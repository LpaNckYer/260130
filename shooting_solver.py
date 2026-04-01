import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm, lstsq
from scipy.integrate import solve_ivp


VAR_ORDER = ["T", "t", "fs", "fl", "x", "y", "w", "rhob", "p"]


def _get_shooting_logger(*, name="shooting_solver", log_path=None, level=logging.INFO):
    """
    Create a dedicated logger for shooting method.
    - Does NOT modify root logger.
    - Avoids duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if log_path is None:
        # Defer file creation until we know the case_name; caller can override.
        log_path = Path("logs") / "shooting_solver.log"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicating handlers
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename) == log_path:
                    return logger
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _project_guess(s):
    """
    Project [T0, x0, y0, w0] into a physically reasonable region.
    This is a numerical safeguard only (not a model constraint proof).
    """
    s = np.asarray(s, dtype=float).copy()
    # T
    s[0] = float(np.clip(s[0], 500.0, 2500.0))
    # x,y,w and sum constraint (keep <= 0.47 like model clamp)
    x = float(np.clip(s[1], 1e-10, 0.47))
    y = float(np.clip(s[2], 0.0, 0.47 - x))
    w = float(np.clip(s[3], 0.0, 0.47 - x - y))
    s[1], s[2], s[3] = x, y, w
    return s


def _pack_y0(params, s):
    """Build y(z=H0) in VAR_ORDER."""
    T0, x0, y0, w0 = s
    return np.array(
        [
            T0,
            params.t_in,
            params.fs_in,
            params.fl_in,
            x0,
            y0,
            w0,
            params.rhob_in,
            params.p_in,
        ],
        dtype=float,
    )


def solve_blast_furnace_by_shooting(
    model,
    *,
    HH=None,
    s0=None,
    tol=1e-6,
    max_iter=25,
    fd_eps=1e-5,
    damping_min=1e-3,
    freeze_ds_eps0=1e-3,
    freeze_ds_eps_min=1e-6,
    freeze_alpha=1.0,
    max_step=0.05,
    method="BDF",
    rtol=1e-5,
    atol=1e-8,
    lm_lambda0=1e-3,
    lm_lambda_max=1e6,
    logger=None,
    log_path=None,
    raise_on_fail=True,
    progress_callback=None,
    save_csv=True,
    output_csv=None,
):
    """
    Shooting method for this 9D blast furnace ODE BVP.

    Known at z=H0 (top): t, fs, fl, rhob, p (in FurnaceParameters: *_in)
    Unknown guessed at z=H0: T0, x0, y0, w0
    Targets at z=HH (bottom): T, x, y, w must match params.T_in, x_in, y_in, w_in

    Returns:
        result dict with converged in/out and profiles.
    """
    params = model.params
    H0 = float(getattr(params, "H0", 0.0))
    HH = float(params.HH if HH is None else HH)

    if logger is None:
        if log_path is None:
            log_path = Path("logs") / f"shooting_{getattr(params, 'case_name', 'case')}.log"
        logger = _get_shooting_logger(log_path=log_path)

    if s0 is None:
        # Use existing initial guess table if available; otherwise fall back.
        if hasattr(params, "value0") and len(params.value0) >= 7:
            s0 = [params.value0[0], params.value0[4], params.value0[5], params.value0[6]]
        else:
            s0 = [800.0, params.x_in, max(params.y_in, 1e-6), params.w_in]
    s = _project_guess(s0)

    target = np.array([params.T_in, params.x_in, params.y_in, params.w_in], dtype=float)
    logger.info(
        "Start shooting: case=%s, z=[%.6g, %.6g], tol=%g, max_iter=%d, method=%s, max_step=%g, rtol=%g, atol=%g",
        getattr(params, "case_name", "case"),
        H0,
        HH,
        tol,
        max_iter,
        method,
        max_step,
        rtol,
        atol,
    )
    logger.info("Initial guess s0=[T0,x0,y0,w0]=%s", s.tolist())
    logger.info("Targets at HH: [T,x,y,w]=%s", target.tolist())

    def ode(z, y):
        Y = y.reshape(9, 1)
        dY = model.blast_furnace_bvp(np.array([z], dtype=float), Y)[:, 0]
        return dY

    def integrate(s_vec):
        y0 = _pack_y0(params, s_vec)
        try:
            sol = solve_ivp(
                ode,
                t_span=(H0, HH),
                y0=y0,
                method=method,
                max_step=max_step,
                rtol=rtol,
                atol=atol,
                dense_output=False,
            )
        except Exception as e:
            raise RuntimeError(f"IVP integration exception: {e}") from e
        if not sol.success or sol.y.shape[1] < 2:
            raise RuntimeError(f"IVP integration failed: {sol.message}")
        yH = sol.y[:, -1]
        F = np.array([yH[0], yH[4], yH[5], yH[6]], dtype=float) - target
        return sol, F

    history = []
    best_F_norm = None

    def _failure_result(message: str, *, last_sol=None, last_F=None, last_F_norm=None):
        return {
            "success": False,
            "case_name": getattr(params, "case_name", "case"),
            "H0": H0,
            "HH": HH,
            "s0_current": {"T0": float(s[0]), "x0": float(s[1]), "y0": float(s[2]), "w0": float(s[3])},
            "target_at_HH": {"T": params.T_in, "x": params.x_in, "y": params.y_in, "w": params.w_in},
            "best_F_norm": best_F_norm,
            "last_F": last_F.tolist() if isinstance(last_F, np.ndarray) else last_F,
            "last_F_norm": last_F_norm,
            "history": history,
            "error": message,
            "output_csv": None,
        }

    # Initial residual
    try:
        sol, F = integrate(s)
    except Exception as e:
        logger.error("Initial integration failed: %s", str(e))
        if raise_on_fail:
            raise
        return _failure_result(f"Initial integration failed: {e}")

    F_norm = float(norm(F))
    best_F_norm = F_norm
    F0_norm = max(F_norm, 1e-30)
    logger.info("Initial residual F=[dT,dx,dy,dw]=%s, ||F||=%.6e", F.tolist(), F_norm)
    # Record iteration-0 residual for downstream traces/plots
    history.append({"iter": 0, "s": s.copy(), "F": F.copy(), "F_norm": F_norm})
    if callable(progress_callback):
        try:
            progress_callback(0, F_norm)
        except Exception:
            pass

    for k in range(1, max_iter + 1):
        history.append({"iter": k, "s": s.copy(), "F": F.copy(), "F_norm": F_norm})
        if callable(progress_callback):
            try:
                progress_callback(k, F_norm)
            except Exception:
                pass
        if F_norm < tol:
            logger.info("Converged at iter=%d, ||F||=%.6e", k, F_norm)
            break
        logger.info("Iter=%d, s=%s, F=%s, ||F||=%.6e", k, s.tolist(), F.tolist(), F_norm)

        freeze_eps_k = max(
            float(freeze_ds_eps_min),
            float(freeze_ds_eps0) * float((F_norm / F0_norm) ** float(freeze_alpha)),
        )
        logger.info(
            "Adaptive freeze eps: eps_k=%.3e (eps0=%.3e, eps_min=%.3e, alpha=%g, ratio=%.3e)",
            freeze_eps_k,
            float(freeze_ds_eps0),
            float(freeze_ds_eps_min),
            float(freeze_alpha),
            float(F_norm / F0_norm),
        )

        # Finite-difference Jacobian J = dF/ds (4x4)
        J = np.zeros((4, 4), dtype=float)
        for j in range(4):
            step = fd_eps * (abs(s[j]) + 1.0)
            sp = s.copy()
            sp[j] += step
            sp = _project_guess(sp)
            _, Fp = integrate(sp)
            J[:, j] = (Fp - F) / (sp[j] - s[j] if sp[j] != s[j] else step)

        # Newton step: J * ds = -F
        # Use least squares for robustness if J is ill-conditioned.
        ds_newton_full, *_ = lstsq(J, -F, rcond=None)
        # Freeze tiny-update variables: remove them from the linear solve.
        active = np.abs(ds_newton_full) >= freeze_eps_k
        if not np.any(active):
            logger.info(
                "All |ds| < freeze_eps_k=%.3e; stopping updates this iteration.",
                freeze_eps_k,
            )
            # No meaningful direction; treat as stalled.
            break

        if not np.all(active):
            J_a = J[:, active]
            ds_a, *_ = lstsq(J_a, -F, rcond=None)
            ds_newton = np.zeros_like(ds_newton_full)
            ds_newton[active] = ds_a
            frozen_idx = [i for i in range(4) if not active[i]]
            logger.info("Freeze vars idx=%s (|ds|<%.3e)", frozen_idx, freeze_eps_k)
        else:
            ds_newton = ds_newton_full

        logger.info("Newton ds=%s", ds_newton.tolist())

        # Damped update (backtracking) with LM fallback if needed.
        def try_step(ds):
            alpha = 1.0
            while alpha >= damping_min:
                s_try = _project_guess(s + alpha * ds)
                try:
                    sol_try, F_try = integrate(s_try)
                except Exception as e:
                    logger.debug("alpha=%.3g integration failed: %s", alpha, str(e))
                    alpha *= 0.5
                    continue
                if norm(F_try) < F_norm:
                    logger.info("Accepted step alpha=%.3g, s_try=%s, ||F_try||=%.6e", alpha, s_try.tolist(), float(norm(F_try)))
                    return True, s_try, sol_try, F_try
                alpha *= 0.5
            return False, None, None, None

        accepted, s_try, sol_try, F_try = try_step(ds_newton)

        if not accepted:
            # Levenberg–Marquardt (modified Newton): (J^T J + λI) ds = -J^T F
            # Apply the same active-variable reduction in LM space.
            JTJ = (J[:, active].T @ J[:, active]) if not np.all(active) else (J.T @ J)
            JTF = (J[:, active].T @ F) if not np.all(active) else (J.T @ F)
            lam = float(lm_lambda0)
            while lam <= lm_lambda_max and not accepted:
                A = JTJ + lam * np.eye(JTJ.shape[0])
                ds_lm_a, *_ = lstsq(A, -JTF, rcond=None)
                if np.all(active):
                    ds_lm = ds_lm_a
                else:
                    ds_lm = np.zeros_like(ds_newton_full)
                    ds_lm[active] = ds_lm_a
                logger.info("LM lambda=%.3g, ds=%s", lam, ds_lm.tolist())
                accepted, s_try, sol_try, F_try = try_step(ds_lm)
                lam *= 10.0

        if not accepted:
            logger.error("Failed to find decreasing step at iter=%d, ||F||=%.6e", k, F_norm)
            if raise_on_fail:
                raise RuntimeError(
                    "Update failed to decrease residual (Newton + LM). "
                    f"Last ||F||={F_norm:.3e}."
                )
            return _failure_result(
                "Update failed to decrease residual (Newton + LM).",
                last_sol=sol,
                last_F=F,
                last_F_norm=F_norm,
            )

        s, sol, F = s_try, sol_try, F_try
        F_norm = float(norm(F))
        if best_F_norm is None or F_norm < best_F_norm:
            best_F_norm = F_norm

    if F_norm >= tol:
        logger.error("Not converged after %d iters, ||F||=%.6e", max_iter, F_norm)
        if raise_on_fail:
            raise RuntimeError(f"Shooting did not converge: ||F||={F_norm:.3e} after {max_iter} iterations.")
        return _failure_result(
            f"Not converged after {max_iter} iterations.",
            last_sol=sol,
            last_F=F,
            last_F_norm=F_norm,
        )

    z = sol.t
    Y = sol.y  # (9, n)

    if save_csv:
        if output_csv is None:
            output_csv = f"{params.case_name}_shooting_{H0:.1f}-{HH:.1f}m.csv"
        df = pd.DataFrame(np.vstack((z, Y)).T, columns=["z"] + VAR_ORDER)
        df.to_csv(output_csv, index=False)
        logger.info("Saved profile CSV: %s", output_csv)

    result = {
        "success": True,
        "case_name": params.case_name,
        "H0": H0,
        "HH": HH,
        "s0_converged": {"T0": s[0], "x0": s[1], "y0": s[2], "w0": s[3]},
        "residual_at_HH": {"dT": F[0], "dx": F[1], "dy": F[2], "dw": F[3], "norm": F_norm},
        "best_F_norm": best_F_norm,
        "last_F_norm": F_norm,
        "inlet_at_H0": {
            "t": params.t_in,
            "fs": params.fs_in,
            "fl": params.fl_in,
            "rhob": params.rhob_in,
            "p": params.p_in,
            "T": s[0],
            "x": s[1],
            "y": s[2],
            "w": s[3],
        },
        "target_at_HH": {"T": params.T_in, "x": params.x_in, "y": params.y_in, "w": params.w_in},
        "solution_at_HH": {"T": float(Y[0, -1]), "x": float(Y[4, -1]), "y": float(Y[5, -1]), "w": float(Y[6, -1])},
        "z": z,
        "Y": Y,
        "history": history,
        "output_csv": output_csv if save_csv else None,
    }
    logger.info("Done shooting: s0_converged=%s, residual=%s", result["s0_converged"], result["residual_at_HH"])
    return result

