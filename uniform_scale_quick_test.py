"""
Quick uniform-scale initial guess test for shooting_solver.

All 4 initial guesses are scaled by the same factor:
    s0 = scale * s*

Scales tested (default): 1.25, 1.20, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75
Each run is limited by a per-sample timeout (default 300s = 5min).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from save_load import load_parameters
from furnace_model import NormalizedFurnaceModel
from shooting_solver import solve_blast_furnace_by_shooting


S_STAR = np.array([634.03337, 0.16221, 0.23559, 0.047715], dtype=float)


@dataclass(frozen=True)
class RunConfig:
    tol: float = 1e-3
    max_iter: int = 25
    max_step: float = 0.05
    rtol: float = 1e-5
    atol: float = 1e-8
    freeze_ds_eps0: float = 1e-3
    freeze_ds_eps_min: float = 1e-6
    freeze_alpha: float = 1.0
    timeout_s: float = 1800.0


def _make_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shooting_uniform_scale")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename) == log_path:
                    return logger
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _make_null_logger() -> logging.Logger:
    logger = logging.getLogger("shooting_uniform_null")
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def _worker(queue, case_name: str, s0: np.ndarray, cfg: RunConfig):
    params = load_parameters(case_name)
    model = NormalizedFurnaceModel(params)
    null_logger = _make_null_logger()
    trace = []

    def cb(it, fnorm):
        trace.append(float(fnorm))
        try:
            queue.put(("trace", float(fnorm)))
        except Exception:
            pass

    res = solve_blast_furnace_by_shooting(
        model,
        s0=s0.tolist(),
        tol=cfg.tol,
        max_iter=cfg.max_iter,
        max_step=cfg.max_step,
        rtol=cfg.rtol,
        atol=cfg.atol,
        freeze_ds_eps0=cfg.freeze_ds_eps0,
        freeze_ds_eps_min=cfg.freeze_ds_eps_min,
        freeze_alpha=cfg.freeze_alpha,
        save_csv=False,
        logger=null_logger,
        raise_on_fail=False,
        progress_callback=cb,
    )
    res["_trace"] = trace
    queue.put(("done", res))


def _run_one(case_name: str, scale: float, cfg: RunConfig) -> Dict:
    s0 = S_STAR * float(scale)
    t0 = time.time()

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, case_name, s0, cfg), daemon=True)
    p.start()
    trace_vals = []
    deadline = time.time() + cfg.timeout_s
    done_res = None

    while time.time() < deadline and p.is_alive():
        try:
            msg = q.get(timeout=0.5)
        except Exception:
            continue
        if isinstance(msg, tuple) and msg and msg[0] == "trace":
            trace_vals.append(float(msg[1]))
        elif isinstance(msg, tuple) and msg and msg[0] == "done":
            done_res = msg[1]
            break

    # drain remaining quickly if finished
    if done_res is None and not p.is_alive():
        while not q.empty():
            msg = q.get()
            if isinstance(msg, tuple) and msg and msg[0] == "trace":
                trace_vals.append(float(msg[1]))
            elif isinstance(msg, tuple) and msg and msg[0] == "done":
                done_res = msg[1]

    if done_res is not None:
        p.join(5)
        if not trace_vals:
            trace_vals = [float(x) for x in done_res.get("_trace", [])]
        trace_json = json.dumps(trace_vals, ensure_ascii=False)
        dt = time.time() - t0
        last = done_res["residual_at_HH"]["norm"] if done_res.get("success") else done_res.get("last_F_norm")
        return {
            "scale": scale,
            "success": bool(done_res.get("success", False)),
            "best_F_norm": done_res.get("best_F_norm", done_res.get("last_F_norm")),
            "last_F_norm": last,
            "residual_trace": trace_json,
            "elapsed_s": dt,
            "error": None if done_res.get("success") else done_res.get("error"),
            "s0_T0": float(s0[0]),
            "s0_x0": float(s0[1]),
            "s0_y0": float(s0[2]),
            "s0_w0": float(s0[3]),
        }

    # timeout
    if p.is_alive():
        p.terminate()
        p.join(5)

    trace_json = json.dumps(trace_vals, ensure_ascii=False)
    dt = time.time() - t0
    return {
        "scale": scale,
        "success": False,
        "best_F_norm": None,
        "last_F_norm": None,
        "residual_trace": trace_json,
        "elapsed_s": dt,
        "error": "timeout",
        "s0_T0": float(s0[0]),
        "s0_x0": float(s0[1]),
        "s0_y0": float(s0[2]),
        "s0_w0": float(s0[3]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="my_design")
    ap.add_argument("--out", default="shooting_uniform_scale_test.csv")
    ap.add_argument("--log", default="logs/shooting_uniform_scale_test.log")
    ap.add_argument("--timeout_s", type=float, default=300.0, help="Per-sample timeout seconds (default 300s=5min)")
    ap.add_argument(
        "--scales",
        default="1.25,1.20,1.15,1.10,1.05,1.00,0.95,0.90,0.85,0.80,0.75",
        help="Comma-separated scales, evaluated in order.",
    )
    args = ap.parse_args()

    scales: List[float] = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    cfg = RunConfig(timeout_s=args.timeout_s)

    logger = _make_logger(Path(args.log))
    logger.info("=== Uniform scale test start ===")
    logger.info("case=%s out=%s timeout_s=%s", args.case, args.out, args.timeout_s)
    logger.info("s*=%s", S_STAR.tolist())
    logger.info("scales=%s", scales)

    rows = []
    print("Uniform scale test")
    print("s* =", S_STAR.tolist())
    print("scales =", scales)
    print("timeout_s =", cfg.timeout_s)

    for i, sc in enumerate(scales, 1):
        print(f"[{i}/{len(scales)}] scale={sc} ...", flush=True)
        row = _run_one(args.case, sc, cfg)
        rows.append(row)
        logger.info(
            "[%d/%d] scale=%s %s best=%s last=%s dt=%.2fs err=%s",
            i,
            len(scales),
            sc,
            "OK" if row["success"] else "FAIL",
            row["best_F_norm"],
            row["last_F_norm"],
            row["elapsed_s"],
            row["error"],
        )
        logger.info("    residual_trace=%s", row.get("residual_trace"))
        ok = "OK" if row["success"] else "FAIL"
        print(f"    {ok} best={row['best_F_norm']} last={row['last_F_norm']} dt={row['elapsed_s']:.1f}s err={row['error']}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    succ = int(df["success"].sum())
    total = len(df)
    print(f"Success rate: {succ}/{total} = {succ/total:.1%}")
    logger.info("=== Uniform scale test end: success_rate=%d/%d ===", succ, total)


if __name__ == "__main__":
    main()

