import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm, cond
from scipy.integrate import solve_bvp

from sigmoid import smooth_heaviside
from heatcurrent_matrix_n import setAa_n
from heatcurrent_matrix_s import setAa_s
from simple_matrix import setAa_linear_n, setAa_p, setAa_constant_s, setAa_constant_n

from constant import pai, R, R_, g_c, T_std, P_std, eps

class FurnaceModel:
    """高炉计算模型"""

    def __init__(self, parameters):
        self.params = parameters
        self.results = {}

    def run(self):
        """运行模型"""
        logging.info(f"计算中：{self.params.case_name}")

        z_guess, state = self._build_initial_guess()
        y_guess = np.array([state[k] for k in ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']])

        final_sol, history = self._solve_with_decreasing_tol(
            self.blast_furnace_bvp, self.bc, [self.params.H0, self.params.HH], y_guess,
            tol_levels=[1e-1, 1e-2, 1e-3]
        )

        self._log_history(history)
        self._plot_and_save_results(final_sol, z_guess)
        return self.results

    def multi_value_interpolation(self, x_control, y_control, num_output_points=2000):
        """
        使用numpy.interp处理多个控制点
        """
        # 生成输出的x坐标
        x_output = np.linspace(x_control[0], x_control[-1], num_output_points)
        
        # 线性插值得到y值
        y_output = np.interp(x_output, x_control, y_control)
        
        return y_output

    def _build_initial_guess(self):
        """构建初始猜测"""
        H_ctrl = [self.params.H0, self.params.H1, self.params.H2, self.params.H3, self.params.HH]
        y0, y1, y2, y3, yH = self.params.value0, self.params.value1, self.params.value2, self.params.value3, self.params.valueH

        state = {
            'T': self.multi_value_interpolation(H_ctrl, [y0[0], y1[0], y2[0], y3[0], yH[0]], self.params.initial_mesh),
            't': self.multi_value_interpolation(H_ctrl, [y0[1], y1[1], y2[1], y3[1], yH[1]], self.params.initial_mesh),
            'fs': self.multi_value_interpolation(H_ctrl, [y0[2], y1[2], y2[2], y3[2], yH[2]], self.params.initial_mesh),
            'fl': self.multi_value_interpolation(H_ctrl, [y0[3], y1[3], y2[3], y3[3], yH[3]], self.params.initial_mesh),
            'x': self.multi_value_interpolation(H_ctrl, [y0[4], y1[4], y2[4], y3[4], yH[4]], self.params.initial_mesh),
            'y': self.multi_value_interpolation(H_ctrl, [y0[5], y1[5], y2[5], y3[5], yH[5]], self.params.initial_mesh),
            'w': self.multi_value_interpolation(H_ctrl, [y0[6], y1[6], y2[6], y3[6], yH[6]], self.params.initial_mesh),
            'rhob': self.multi_value_interpolation(H_ctrl, [y0[7], y1[7], y2[7], y3[7], yH[7]], self.params.initial_mesh),
            'p': self.multi_value_interpolation(H_ctrl, [y0[8], y1[8], y2[8], y3[8], yH[8]], self.params.initial_mesh),
        }
        z_guess = np.linspace(self.params.H0, self.params.HH, self.params.initial_mesh)
        return z_guess, state

    def _solve_with_decreasing_tol(self, ode, bc, x_span, y_init, tol_levels=None):
        """逐步减小容差求解BVP"""
        if tol_levels is None:
            tol_levels = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

        x = np.linspace(x_span[0], x_span[1], len(y_init[0]))
        history = []

        for i, tol in enumerate(tol_levels):
            logging.info(f"第 {i+1} 轮迭代，容差: {tol}")
            sol = solve_bvp(ode, bc, x, y_init, tol=tol, max_nodes=len(x)*20, verbose=2)

            history.append({
                'tol': tol,
                'solution': sol,
                'success': sol.success,
                'n_nodes': len(sol.x)
            })

            if not sol.success:
                logging.warning(f"第 {i+1} 轮未收敛，继续")
            x = np.linspace(x_span[0], x_span[1], len(sol.x))
            y_init = sol.sol(x)

        return sol, history

    def _log_history(self, history):
        """记录迭代历史"""
        logging.info("=== 迭代历史 ===")
        for i, record in enumerate(history):
            logging.info(f"轮次 {i+1}: 容差={record['tol']:.1e}, 节点数={record['n_nodes']}, 成功={record['success']}")

    def _plot_and_save_results(self, sol, z_guess):
        """绘制并保存结果"""
        y_plot = sol.sol(z_guess)
        variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']

        plt.figure(figsize=(12, 8))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.plot(z_guess, y_plot[i])
            plt.ylabel(variables[i])
            plt.xlabel('z (m)')
        plt.tight_layout()
        plt.close()  # 避免显示，节省内存

        df = pd.DataFrame(np.vstack((z_guess, y_plot)).T, columns=['z'] + variables)
        df.to_csv(f'{self.params.case_name}_{self.params.H0:.1f}-{self.params.HH:.1f}m.csv', index=False)

        self.results = {
            "case_name": self.params.case_name,
            "H0": z_guess[0],
            "HH": z_guess[-1],
            "T_out": y_plot[0, 0],
            "t_out": y_plot[1, -1],
            "fs_out": y_plot[2, -1],
            "fl_out": y_plot[3, -1],
            "x_out": y_plot[4, 0],
            "y_out": y_plot[5, 0],
            "w_out": y_plot[6, 0],
            "rhob_out": y_plot[7, -1],
            "p_bottom": y_plot[8, -1]
        }
    
    # bvp definition
    def blast_furnace_bvp(self,Z,Y):
        """
        Args:
            Z: height. ndarray. (n,)
            Y: state variables (T,t,fs,fl,x,y,w,rho_b,p). ndarray. (m,n)
        Returns:
            dY/dz: space derivative of state variables. ndarray. (m,n)
        """
        m, n = Y.shape
        res = np.empty((m, n))
        for i in range(n):
            z = Z[i]
            T,t,fs,fl,x,y,w,rho_b,p = Y[:,i]
            state = self._clamp_state(T=T, t=t, fs=fs, fl=fl, x=x, y=y, w=w, p=p, rho_b=rho_b)
            res[:, i] = [
                self.dTdz(z, state),
                self.dtdz(z, state),
                self.dfsdz(z, state),
                self.dfldz(z, state),
                self.dxdz(z, state),
                self.dydz(z, state),
                self.dwdz(z, state),
                self.drhobdz(z, state),
                self.dpdz(z, state)
            ]
        return res

    def bc(self,ya,yb):
        """
        Args:
            ya: boundary condition of state variables at z=0. ndarray. (n,)
            yb: boundary condition of state variables at z=H. ndarray. (n,)
        Returns:
            bc: boundary condition. ndarray. (n,)
        """
        return np.array([yb[0]-self.params.T_in,
                        ya[1]-self.params.t_in,
                        ya[2]-self.params.fs_in,
                        ya[3]-self.params.fl_in,
                        yb[4]-self.params.x_in,
                        yb[5]-self.params.y_in,
                        yb[6]-self.params.w_in,
                        ya[7]-self.params.rhob_in,
                        ya[8]-self.params.p_in])

    def _clamp_state(self, **state):
        """统一 clamp 逻辑"""
        # 定义边界字典以提高可维护性
        bounds = {
            'T': (500, 2500),
            't': (400, 2500),
            'fs': (0, 1),
            'fl': (0, 1),
            'x': (1e-10, 0.47),
            'p': (1e4, 3e4),
            'rho_b': (800, 1200),
        }
        
        # 对独立变量进行裁剪
        for key, (min_val, max_val) in bounds.items():
            if key in state:
                state[key] = np.clip(state[key], min_val, max_val)
        
        # 对依赖变量进行裁剪（x, y, w 相互依赖）
        if 'y' in state:
            state['y'] = np.clip(state['y'], 0, 0.47 - state.get('x', 0))
        if 'w' in state:
            state['w'] = np.clip(state['w'], 0, 0.47 - state.get('x', 0) - state.get('y', 0))
        
        return state

    # odes
    def dTdz(self, z, state):
        """differential equation of T
        temperature of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]   
            U (float): overall heat transfer coefficient based on inner surface area of furnace-wall. [kcal / m2 * hr * K]
            T_we (flaot): exit tempareture of cooling water. [K]

        Returns:
            dd (float): [K / m]

        """
        
        T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'],  state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        C,dCdT = self.HeatCapacity_Gas(T,x,y,w) # C (float): heat capacity of gas. [kcal / kg * K] ; dCdT (float): differential of C with T. [kcal / kg * K**2]

        q1 = self.Heat_1() # [kcal / m3 bed * hr]
        q2 = self.Heat_2(z,T,t,fs,fl,x,y,w,p) # [(kmol / m3 bed * hr) * (kg / m3)]
        q3 = self.Heat_3(z,T,t,x,y,w) # [kcal / m3 bed * hr]

        dd = (Az * (q1 + 22.4*C*q2*T + q3) + pai * Dz * self.params.U * (T - self.params.T_we)) / (rho * F * (C + T*dCdT))

        return dd
    
    def dtdz(self,z,state):
        """differential equation of t
        temperature of solid particle

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
            rho_b (float): bulk density of solid particles. [kg / m3 bed]
        Operate:    
            Fs (float): volume rate of solid particles. [m3 bed / hr]

        Returns:
            dd (float): [K / m]
        """
        
        T, t, fs, fl, x, y, w, rho_b, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['rho_b'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        Cs,dCsdt = self.HeatCapacity_Solid(t) # Cs (float): specific heat of solid particles. [kcal / kg * K] ; dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]

        q3 = self.Heat_3(z,T,t,x,y,w) # [kcal / m3 bed * hr]
        q4 = self.Heat_4(z,T,t,fs,fl,x,y,w,p) # [kcal / m3 bed * hr]
        q5 = self.Heat_5(z,T,t,fs,fl,x,y,w,p) # [kg / m3 bed * hr]

        dd = Az * (q3 + Cs*t*q5 + q4) / (rho_b * self.params.Fs * (Cs + t*dCsdt))
        # return np.asarray(dd).item()   
        return dd

    def dfsdz(self, z, state):
        """differential equation of fs
        fractional reduction of iron ore

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            Fs (float): volume rate of solid particles. [m3 bed / hr]
            c_H0 (float): initial concentration of hematite. [kmol / m3 bed]

        Returns:
            dd (float): [1 / m]
        """
        
        T, t, fs, x, y, w, p = state['T'], state['t'], state['fs'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]    

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        
        weight = smooth_heaviside(t-1673,k=5)
        dd1 = Az * (R1 + R5) / 3 / self.params.Fs / self.params.c_H0
        dd2 = Az * (R3 + R5) / 3 / self.params.Fs / self.params.c_H0
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dfldz(self,z,state):
        """differential equation of fl
        fractional decomposition of limestone

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            Fs (float): volume rate of solid particles. [m3 bed / hr]
            c_L0 (float): initial concentration of limestone. [kmol / m3 bed]

        Returns:
            dd (float): [1 / m]
        """
        
        T, t, fl, x, y, w, p = state['T'], state['t'], state['fl'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]   

        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]

        dd = Az * R4 / self.params.Fs / self.params.c_L0
        return dd

    def dxdz(self,z,state):
        """differential equation of x
        molar fraction of CO in bulk of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Returns:
            dd (float): [1 / m]
        """
        
        T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1200,k=1)
        R21 = R2
        R22 = R2 + R1+R4+R7
        R2 = (1-weight)*R21 + weight*R22

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = 22.4 * Az * ((1+0*x)*R1 + (x-2)*R2 + x*R4 + (x-1)*R6 + R7) / F
        dd2 = 22.4 * Az * ((-1+1*x)*R3 + (x-2)*R2 + x*R4 + (x-1)*R6 + R7) / F
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dydz(self,z,state):
        """differential equation of y
        molar fraction of CO2 in bulk of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]
        
        Returns:
            dd (float): [1 / m]
        """
        
        T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1200,k=1)
        R21 = R2
        R22 = R2 + R1+R4+R7
        R2 = (1-weight)*R21 + weight*R22

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = 22.4 * Az * ((0*y-1)*R1 + (y+1)*R2 + (y-1)*R4 + y*R6 - R7) / F
        dd2 = 22.4 * Az * ((1*y-0)*R3 + (y+1)*R2 + (y-1)*R4 + y*R6 - R7) / F
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dwdz(self,z,state):
        """differential equation of w
        molar fraction of H2 in bulk of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]
        
        Returns:
            dd (float): [1 / m]
        """
        
        T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
       
        weight = smooth_heaviside(t-1200,k=1)
        R21 = R2
        R22 = R2 + R1+R4+R7
        R2 = (1-weight)*R21 + weight*R22

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = 22.4 * Az * (0*w*R1 + w*R2 + w*R4 + R5 + (w-1)*R6 - R7) / F
        dd2 = 22.4 * Az * (1*w*R3 + w*R2 + w*R4 + R5 + (w-1)*R6 - R7) / F
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def drhobdz(self,z,state):
        """differential equation of rho_b
        bulk density of solid particles

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): tempareture of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:    
            Fs (float): volume rate of solid particles. [m3 bed / hr]
        
        Returns:
            dd (float): [kg / m3 bed * m]
        """
         
        T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1200,k=1)
        R21 = R2
        R22 = R2 + R1+R4+R7
        R2 = (1-weight)*R21 + weight*R22

        weight = smooth_heaviside(t-1673,k=5)
        dd1 = -Az * ((16+12*0)*R1 + 12*R2 + 44*R4 + 16*R5 + 12*R6) / self.params.Fs
        dd2 = -Az * ((16+12*1)*R3 + 12*R2 + 44*R4 + 16*R5 + 12*R6) / self.params.Fs
        dd = (1-weight)*dd1 + weight*dd2
        return dd

    def dpdz(self,z,state):
        """differential equation of p
        pressure of gas

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]
        Operate:
            epsilon (float): fractional void in bed. [-]
            F_b (float): volume rate of dry blast. [Nm3 / min]
            
        Returns:
            dd (float): [Kg / m2 * m]
        """
        
        T, x, y, w, p = state['T'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]

        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        G = F * rho / (Az * self.params.epsilon) # G (float): mass velocity of gas. [kg / m2 * hr]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        Re = self.params.d_p * G / miu
        fk = (1.75 + 150 * (1 - self.params.epsilon)) / Re

        dd = fk * (1 - self.params.epsilon) * G**2 * P_std * T / (g_c * self.params.epsilon**3 * self.params.d_p * rho * T_std * p)

        return dd

    # heat
    def Heat_1(self):
        return 0.0
    
    def Heat_2(self,z,T,t,fs,fl,x,y,w,p):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            q (float): [(kmol / m3 bed * hr) * (kg / m3)]
        """
        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]

        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        weight = smooth_heaviside(t-1200,k=1)
        R21 = R2
        R22 = R2 + R1+R4+R7
        R2 = (1-weight)*R21 + weight*R22

        weight = smooth_heaviside(t-1673,k=5)
        q1 = (1.2507*0 + 0.7261*1)*R1 + 0.5246*R2 + 1.9768*R4 + 0.7143*R5 + 0.5364*R6
        q2 = (1.2507*1 + 0.7261*(-1))*R3 + 0.5246*R2 + 1.9768*R4 + 0.7143*R5 + 0.5364*R6
        q = q1 * (1-weight) + q2 * weight

        return q
    
    def Heat_3(self,z,T,t,x,y,w):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
        
        Operate:    
            epsilon (float): fractional void in bed. [-]
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            phi_o (float): shape factor of iron ore. [-]
            d_o (float): average diameter of particles of iron ore. [m]    
            
        Returns:
            q (float): [kcal / m3 bed * hr]
        """
        Dz = self.params.Diameter_BF(z)
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]

        C = self.HeatCapacity_Gas(T,x,y,w)[0] # C (float): specific heat of gas. [kcal / kg * K]

        G = rho * F / Az # G (float): mass velocity of gas. [kg / m2 * hr]
        Re = self.params.d_p * G / miu
        k = 0.06 # k (float): thermal conductivity of gas. [kcal / m * hr * K]
        Pr = C * miu / k
        Nu = 2.0 + 0.60*Re**(1/2)*Pr**(1/3)
        h_p = Nu * k / self.params.d_p # h_p (float): particle-to-fluid heat transfer coefficient. [kcal / m2 * hr * K]

        q = 6 * (1-self.params.epsilon) * h_p * (T-t) / self.params.phi_o / self.params.d_p
        # print(f"hp={h_p}")
        # print(f"q3={q}")
        return q
    
    def Heat_4(self,z,T,t,fs,fl,x,y,w,p):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            q (float): [kcal / m3 bed * hr]
        """
        # if fs < 0.111:
        #     H1 = -7.88e3 # [kcal / kmol CO]
        #     H5 = -2.8e3 # [kcal / kmol H2]
        # elif fs < 0.333:
        #     H1 = 7.12e3
        #     H5 = 16.1e3 
        # else:
        #     H1 = -5.45e3
        #     H5 = 6.5e3

        weight1 = smooth_heaviside(fs-0.111,k=200)
        weight2 = smooth_heaviside(fs-0.333,k=200)

        H1 = np.zeros_like(z)
        H5 = np.zeros_like(z)
        mask = (fs < 0.222)
        H1[mask] = (1-weight1[mask])*-7.88e3 * 1/9 + weight1[mask]*7.12e3 * 2/9
        H5[mask] = (1-weight1[mask])*-2.8e3 * 1/9 + weight1[mask]*16.1e3 * 2/9
        H1[~mask] = (1-weight2[~mask])*7.12e3 * 2/9 + weight2[~mask]*-5.45e3 * 2/3
        H5[~mask] = (1-weight2[~mask])*16.1e3 * 2/9 + weight2[~mask]*6.5e3 * 2/3
        
        H2 = 40.8e3 # [kcal / kmol CO2]
        H3 = 31.13e3 # [kcal / kmol CO]
        H4 = 42.5e3 # [kcal / kmol CO2]
        H6 = 31.5e3 # [kcal / kmol CO]
        H7 = -9.84e3 # [kcal / kmol CO2]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        weight = smooth_heaviside(t-1200,k=1)
        R21 = R2
        R22 = R2 + R1+R4+R7
        R2 = (1-weight)*R21 + weight*R22

        weight = smooth_heaviside(t-1673,k=2)
        q1 = -H1*R1 -H2*R2 -H4*R4 -H5*R5 -H6*R6 -H7*R7
        q2 = -H3*R3 -H2*R2 -H4*R4 -H5*R5 -H6*R6 -H7*R7
        q = (1-weight)*q1 + weight*q2

        return q    
    
    def Heat_5(self,z,T,t,fs,fl,x,y,w,p):
        """
        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            q (float): [kg / m3 bed * hr]
        """
        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]

        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        weight = smooth_heaviside(t-1200,k=1)
        R21 = R2
        R22 = R2 + R1+R4+R7
        R2 = (1-weight)*R21 + weight*R22

        weight = smooth_heaviside(t-1673,k=5)
        q1 = 16*R1 + 12*R2 + 44*R4 + 16*R5 + 12*R6
        q2 = 28*R3 + 12*R2 + 44*R4 + 16*R5 + 12*R6
        q = q1 * (1-weight) + q2 * weight
        # q = np.zeros_like(q)
        return q

    # reaction_rate
    def ReactionRate_1(self,z,T,t,fs,x,y,w,p):
        """overall reaction rate per unit volume of bed in reaction
        1/3 Fe2O3 + CO = 2/3 Fe + CO2

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            d_o (float): average diameter of particles of iron ore. [m]
            phi_o (float): shape factor of iron ore. [-]
            N_o (int): number of particles of iron ore per unit volume of bed. [1 / m3 bed]
            epsilon_o (float): porosity of iron ore. [-]
            
        Returns:
            r (float): reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        
        Raises:
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]

        D_CO = self.DiffusionCoefficient_CO(t,p) # D_CO (float): diffusion coefficient of CO in blast furnace gas. [m2 / hr]

        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_o * u * rho / miu

        Sc = miu / rho / D_CO
        Sh = 2.0 + 0.55*Re**(1/2)*Sc**(1/3)
        kf = self.TransferCoefficient_Gas(Sh,D_CO,self.params.d_o) # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

        epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
        xi = 0.238 * self.params.epsilon_o + 0.04
        Ds = D_CO * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of CO in reduced iron phase. [m2 / hr]

        k = 347 * np.exp(-3460/t) # k (float): rate constant of reaction. [m / hr]

        K = self.smooth_R1(t,fs) # K (float): equilibrium constant of reaction. [-]

        xe = (x+y) / (1+K)

        r = pai * self.params.d_o**2 * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 * (x-xe) / 22.4 / t / (1/kf + self.params.d_o/2*((1-fs+eps)**(-1/3) - 1)/Ds + ((1-fs+eps)**(2/3)*k*(1+1/K))**(-1))
        return r
    
    def ReactionRate_2(self,z,T,t,fs,x,y,w,p):
        """overall reaction rate per unit volume of bed in reaction
        C + CO2 = 2CO

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            d_c (float): average diameter of particles of coke. [m]
            phi_c (float): shape factor of coke. [-]
            N_c (int): number of particles of coke per unit volume of bed. [1 / m3 bed]    
            rho_pc (float): apparent density of coke. [kg / m3 coke]

        Returns:
            r (float): reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]

        Raises:
        """
        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]

        D_CO2 = self.DiffusionCoefficient_CO2(t,p)    # D_CO2 (float): diffusion coefficient of CO2 in blast furnace gas. [m2 / hr]
        Re = self.params.d_c * u * rho / miu
        Sc = miu / rho / D_CO2
        Sh = 2.0 + 0.55*Re**(1/2)*Sc**(1/3)
        kf = self.TransferCoefficient_Gas(Sh,D_CO2,self.params.d_c)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

        k = 2.99e13 * np.exp(-80000/R/t) # k (float): rate constant of reaction. [m3 / kg*hr]
        
        epsilon_c = 0.45
        xi_c = 0.238 * epsilon_c + 0.04
        Ds = D_CO2 * epsilon_c * xi_c # Ds (float): intraparticle diffusion coefficient of CO2 in reduced iron phase. [m2 / hr]

        m = (self.params.d_c/2) * (self.params.rho_pc*k/Ds)**(1/2)
        Ef = 3 * (m*self.stable_coth(m) - 1) / m**2 # Ef (float): effectiveness factor. [-]

        r2 = pai * self.params.d_c**2 * self.params.phi_c**(-1) * self.params.N_c * 273 * (p/P_std) * y / 22.4 / t / (1/kf + 6/(self.params.d_c*self.params.rho_pc*Ef*k+eps))
        return r2  

    def ReactionRate_3(self,t,fs):
        """overall reaction rate per unit volume of bed in reaction
        FeO(l) + C(s) = Fe(l) + CO(g)

        Args:
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-] 

        Operate:
            F_mH (float): molar rate of flow of hematite. [kmol / hr]
            W_S (float): mass rate of flow of slag. [kg / hr]
            rho_S (float): density of slag. [kg / m3]
            rho_W (float): density of wustite. [kg / m3]

        Materials:
            epsilon_c (float): porosity of coke. [-]
            phi_c (float): shape factor of coke. [-]
            d_c (float): average diameter of particles of coke. [m]

        Returns:
            r (float): reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]

        Raises:
        """
        k = 4.66e4 * np.exp(-53300/R/t) # k (float): rate constant of reaction. [m4 / kmol CO * hr]

        rho_P = 8540 - 0.750*t  # rho_P (float): density of pig iron. [kg / m3]
        c = 3 * self.params.F_mH * (1-fs+eps) / ((self.params.W_S/self.params.rho_S) + 71.85*(3*self.params.F_mH*(1-fs+eps))/self.params.rho_W + 55.85*(3*fs-1)*self.params.F_mH/rho_P) # c (float): concentration of wustite in molten slag. [kmol FeO / m3 slag]

        beta = 0.078    # effective surface area for reaction per unit surface area of coke particles
        r = k * beta * (6*(1-self.params.epsilon_c) / self.params.phi_c / self.params.d_c) * c**2

        return r

    def ReactionRate_4(self,z,T,t,fl,x,y,w,p):
        """overall reaction rate per unit volume of bed in reaction
        CaCO3 = CaO + CO2

        Args:
            z (float): height from the stock line. [m]    
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fl (float): fractional decomposition of limestone. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            phi_L (float): shape factor of limestone. [-]
            N_L (int): number of particles of limestone per unit volume of bed. [1 / m3 bed]
            d_L (float): average diameter of particles of limestone. [m]  

        Returns:
            r (float): reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]

        Raises:
        """

        D_CO2 = self.DiffusionCoefficient_CO2(t,p)    # D_CO2 (float): diffusion coefficient of CO2 in blast furnace gas. [m2 / hr]
        epsilon_pL = 0.20
        epsilon_vL = 0.702*epsilon_pL + 0.298
        xi_L = epsilon_vL**0.41
        Ds = D_CO2 * epsilon_vL * xi_L # Ds (float): intraparticle diffusion coefficient of CO2 in phase of CaO. [m2 / hr]

        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]    
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_L * u * rho / miu
        Sc = miu / rho / D_CO2
        Sh = 2.0 + 0.55*Re**(1/2)*Sc**(1/3)
        kf = self.TransferCoefficient_Gas(Sh,D_CO2,self.params.d_L) # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

        k = 5.47e6 * np.exp(-40000/R/T) # k (float): rate constant of reaction. [kmol CO2 / m2 * hr]

        K = np.exp(-8202.5/t + 7.0099) # K (float): equilibrium constant of reaction. [atm]

        Fm = F / 22.4 # Fm (float): molar flow rate of gas. [kmol / hr]
        Fr = F * T/T_std * P_std/p # Fr (float): real volume rate of flow of gas. [m3 /hr]
        c = Fm / Fr # c (float): concentration of gas. [kmol / m3]
        ce = K / R_ / T # ce (float): concentration of CO2 at equilibrium. [kmol CO2 / m3]

        r = np.zeros_like(z)

        with np.errstate(invalid='raise'):  # 将无效值错误转为异常
            r[~(fl>=1)] = pai * self.params.d_L**2 * self.params.phi_L**(-1) * self.params.N_L * (ce[~(fl>=1)]-c[~(fl>=1)]*y[~(fl>=1)]) / (1/kf[~(fl>=1)] + self.params.d_L*((1-fl[~(fl>=1)]+eps)**(-1/3) - 1)/2/Ds[~(fl>=1)] + ((1-fl[~(fl>=1)]+eps)**(2/3)*k[~(fl>=1)]*R_*t[~(fl>=1)]/(K[~(fl>=1)]+eps))**(-1))
            r[(fl>=1)] = 0
            
        np.clip(r, 0, None, out=r)           
        return r

    def ReactionRate_5(self,z,T,t,fs,x,y,w,p):
        """overall reaction rate per unit volume of bed in reaction
        1/3 Fe2O3 + H2 = 2/3 Fe + H2O

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            fs (float): fractional reduction of iron ore. [-]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            d_o (float): average diameter of particles of iron ore. [m]
            phi_o (float): shape factor of iron ore. [-]
            epsilon_o (float): porosity of iron ore. [-]
            N_o (int): number of particles of iron ore per unit volume of bed. [1 / m3 bed]

        Returns:
            r (float): reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        Raises:
        """

        D_H2 = 3.960E-6*t**1.78 / (p/P_std)    # D_H2 (float): diffusion coefficient of H2 in blast furnace gas. [m2 / hr]
        epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
        xi = 0.238 * self.params.epsilon_o + 0.04
        Ds = D_H2 * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of H2 in reduced iron phase. [m2 / hr]

        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        v = self.MolarFaction_H2O(x,y,w) # v: molar fraction of H2O in bulk of gas. [-]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_o * u * rho / miu
        Sc = miu / rho / D_H2
        Sh = 2.0 + 0.55*Re**(1/2)*Sc**(1/3)
        kf = self.TransferCoefficient_Gas(Sh,D_H2,self.params.d_o)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]
        k,K = self.smooth_R5(t)  # smoothed k,K

        we = (w + v) / (1+K) # we (float): molar fraction of H2 at equilibrium. [-]

        r = np.zeros_like(z)

        with np.errstate(invalid='raise'):  # 将无效值错误转为异常
            r[~(fs>=1)] = pai * self.params.d_o**(2) * self.params.phi_o**(-1) * self.params.N_o * 273 * (p[~(fs>=1)]/P_std) * (w[~(fs>=1)]-we[~(fs>=1)]) / 22.4 / t[~(fs>=1)] / (1/kf[~(fs>=1)] + self.params.d_o/2*((1-fs[~(fs>=1)]+eps)**(-1/3) - 1)/Ds[~(fs>=1)] + ((1-fs[~(fs>=1)]+eps)**(2/3)*k[~(fs>=1)]*(1+1/K[~(fs>=1)]))**(-1))
            r[(fs>=1)] = 0

        np.clip(r, 0, None, out=r)
        return r

    def ReactionRate_6(self,z,T,t,x,y,w,p):
        """overall reaction rate per unit volume of bed in reaction
        C + H2O = CO + H2

        Args:
            z (float): height from the stock line. [m]
            T (float): temperature of gas. [K]
            t (float): temperature of solid particles(molten materials). [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Operate:
            F_b (float): volume rate of dry blast. [Nm3 / min]

        Materials:
            d_c (float): average diameter of particles of coke. [m]
            phi_c (float): shape factor of coke. [-]
            N_c (int): number of particles of coke per unit volume of bed. [1 / m3 bed]
            rho_pc (float): apparent density of coke. [kg / m3 coke]

        Returns:
            r (float): reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]

        Raises:
        """
        D_H2O = 1.500E-6*t**1.78 / (p/P_std)    # D_H2O (float): diffusion coefficient of H2O in blast furnace gas. [m2 / hr]
        Ds = D_H2O * 6.620e-2 # Ds (float): intraparticle diffusion coefficient of H2O in reduced iron phase. [m2 / hr]

        Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        v = self.MolarFaction_H2O(x,y,w) # v: molar fraction of H2O in bulk of gas. [-]
        u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        Re = self.params.d_c * u * rho / miu
        Sc = miu / rho / D_H2O
        Sh = 2.0 + 0.55*Re**(1/2)*Sc**(1/3)
        kf = self.TransferCoefficient_Gas(Sh,D_H2O,self.params.d_c)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]    

        k = 4.83e4 * t * np.exp(-17311/t) # k (float): rate constant of reaction. [m3 / kg * hr]

        m_ = (self.params.d_c/2) * (self.params.rho_pc*k/Ds)**(1/2)
        Ef_ = 3 * (m_*self.stable_coth(m_) - 1) / m_**2 # Ef_ (float): effectiveness factor. [-]

        r = pai * self.params.d_c**(2) * self.params.phi_c**(-1) * self.params.N_c * 273 * (p/P_std) * v /22.4 / t / (1/kf + 6/(self.params.d_c*self.params.rho_pc*Ef_*k+eps))
        return r

    def ReactionRate_7(self,T,x,y,w,p):
        """change in moles of H2 caused by reaction
        CO + H2O = CO2 + H2

        Args:  
            T (float): temperature of gas. [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            r (float): change in moles of H2. [kmol H2 / m3 bed * hr]

        Raises:
        """
        v = self.MolarFaction_H2O(x,y,w) # v: molar fraction of H2O in bulk of gas. [-]

        r_forward = 7.29e11 * x**(1/2) * v * (p/P_std/T)**(3/2) * self.params.epsilon * np.exp(-67300/R/T) / np.sqrt(1+14.158*w*p/P_std/T)
        r_backward = 1.386e10 * y * w**(1/2) * (p/P_std/T)**(3/2) * self.params.epsilon *np.exp(-57000/R/T) / (1+4.247*x*p/P_std/T)
        r = r_forward - r_backward
        return r    
    
    # 辅助函数
    def stable_coth(self, m):
        m = np.asarray(m)
        abs_m = np.abs(m)
        
        result = np.zeros_like(abs_m)
        mask = (abs_m > 700)
        result[mask] = np.sign(m[mask]) * 1.0
        result[~mask] = np.cosh(m[~mask]) / np.sinh(m[~mask])
        return result

        
    def smooth_R5(self, t,t0=848,k=10):
        weight = smooth_heaviside(t - t0, k=2)

        k1 = 102.78 * t * np.exp(-14900/R/t) # k (float): rate constant of reaction. [m / hr]
        K1 = np.exp(8.883 - 8475/t) # K (float): equilibrium constant of reaction. [-]

        k2 = 82.50 * t * np.exp(-15300/R/t)  
        K2 = np.exp(1.0837 - 1737.2/t)

        k = (1 - weight) * k1 + weight * k2
        K = (1 - weight) * K1 + weight * K2
        # return k,(K+eps)
        return k,K

    def smooth_R1(self, t,fs,t0=848,fs0=0.111,fs1=0.333,k=10):
        t = np.asarray(t)
        fs = np.asarray(fs)
        weight1 = smooth_heaviside(t - t0, k=20)
        weight2 = smooth_heaviside(fs - fs0, k=200)
        weight3 = smooth_heaviside(fs - fs1, k=200)

        K11 = np.exp(4.91 + 6235/t)
        K12 = np.exp(-0.7625 + 543.3/t)
        K21 = np.exp(4.91 + 6235/t)
        K22 = np.exp(2.13 - 2050/t)
        K23 = np.exp(-2.642 + 2164/t)

        K = np.zeros_like(t)
        mask1 = (fs < (fs0+fs1)/2)
        K[mask1] = (1 - weight1[mask1]) * ((1 - weight2[mask1]) * K11[mask1] + weight2[mask1] * K12[mask1]) + weight1[mask1] * ((1 - weight2[mask1]) * K21[mask1] + weight2[mask1] * K22[mask1])
        K[~mask1] = (1 - weight1[~mask1]) * K12[~mask1] + weight1[~mask1] * ((1 - weight3[~mask1]) * K22[~mask1] + weight3[~mask1] * K23[~mask1])

        # np.clip(K, None, 1e5, out=K)
        return K    # K (float): equilibrium constant of reaction. [-]
    
    # 简单变量计算函数
    def VolumeRate_Gas(self,x,y):
        """
        Args:
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]

        Returns:
            F (float): volume rate of flow of gas. [Nm3 / hr]

        Raises:
        """
        F = (47.4 * self.params.F_b + self.params.F_0*(self.params.w_0+self.params.v_0)) / (1-x-y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        return F

    def MolarFaction_H2O(self,x,y,w):
        """
        Args:
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]

        Returns:
            v (float): molar fraction of H2O in bulk of gas. [-]

        Raises:
        """
        F = self.VolumeRate_Gas(x,y)
        v = self.params.F_0*(self.params.w_0+self.params.v_0) / F - w # v: molar fraction of H2O in bulk of gas. [-]
        return v
    
    def Density_Gas(self,x,y,w):
        """
        Args:
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]

        Returns:
            rho (float): density of blast furnace gas. [kg / Nm3]

        Raises:
        """
        v = self.MolarFaction_H2O(x,y,w)
        rho = 1.2507 + 0.7261*y - 1.1614*w - 0.4471*v # rho (float): density of blast furnace gas. [kg / Nm3]
        return rho
    
    def Viscosity_Gas(self,T):
        """
        Args:
            T (float): temperature of gas. [K]

        Returns:
            miu (float): viscosity of blast furnace gas. [kg / m * hr]

        Raises:
        """
        miu = 4.960e-3 * T**(3/2) / (T+103) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        return miu
    
    def TransferCoefficient_Gas(self,Sh,D,d):
        """
        Args:
            Sh (float): Schmidt number of gas-film. [-]
            D (float): diffusion coefficient of gas. [m2 / hr]
            d (float): particle diameter. [m]

        Returns:
            kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

        Raises:
        """
        kf = Sh * D / d  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]
        return kf
    
    def DiffusionCoefficient_CO(self,t,p):
        """
        Args:
            t (float): temperature of solid particles(molten materials). [K]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            D_CO (float): diffusion coefficient of CO. [m2 / hr]

        Raises:
        """
        weight = smooth_heaviside(t-848,k=5)
        D_CO_1 = 2.592e-6 * t**(1.78) / (p/P_std)
        D_CO_2 = 2.592e-6 * (t)**(2.0) / (p/P_std)
        D_CO = (1-weight) * D_CO_1 + weight * D_CO_2 # D_CO (float): diffusion coefficient of CO. [m2 / hr]
        return D_CO

    def DiffusionCoefficient_CO2(self,t,p):
        """
        Args:
            t (float): temperature of solid particles(molten materials). [K]
            p (float): pressure of gas. [Kg / m2]

        Returns:
            D_CO2 (float): diffusion coefficient of CO2. [m2 / hr]

        Raises:
        """
        D_CO2 = 2.236E-6 * t**(1.78) / (p/P_std)    # D_CO2 (float): diffusion coefficient of CO2 in blast furnace gas. [m2 / hr]
        return D_CO2
    
    def HeatCapacity_Gas(self,T,x,y,w):
        """
        Args:
            T (float): temperature of gas. [K]
            x (float): molar fraction of CO in bulk of gas. [-]
            y (float): molar fraction of CO2 in bulk of gas. [-]
            w (float): molar fraction of H2 in bulk of gas. [-]

        Returns:
            C (float): heat capacity of gas. [kcal / kg * K]
            dCdT (float): specific heat of gas differential T. [kcal / kg * K**2]

        Raises:
        """
        v = self.MolarFaction_H2O(x,y,w)
        S1 = 6.50 + 0.10*x + 4.00*y + 0.12*w + 0.66*v 
        S2 = (1.00 + 0.20*x + 1.40*y - 0.19*w + 1.58*v)*1e-3
        M = 28 + 16*y - 26*w - 10*v
        C = (S1 + S2*T - 2.00e5*y/T**2) / M # C (float): specific heat of gas. [kcal / kg * K]
        dCdT = (S2 + 4e5*y/T**3) / M # dCdT (float): specific heat of gas differential T. [kcal / kg * K**2]
        return C,dCdT
    
    def HeatCapacity_Solid(self,t):
        """
        Args:
            t (float): temperature of solid particles(molten materials). [K]

        Returns:
            Cs (float): specific heat of solid particles. [kcal / kg * K]
            dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]

        Raises:
        """
        Cs = 0.1897 + 3.147e-5 * t # Cs (float): specific heat of solid particles. [kcal / kg * K]
        dCsdt = 3.147e-5 # dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]
        return Cs,dCsdt


class NormalizedFurnaceModel(FurnaceModel):
    """归一化版本的高炉模型"""
    
    def __init__(self, parameters):
        super().__init__(parameters)
        self.norms = self._compute_normalization_factors()
    
    def _compute_normalization_factors(self):
        """基于边界条件计算归一化因子"""
        return {
            'T': max(self.params.T_in, 1000),
            't': max(self.params.t_in, 1000),
            'fs': 1.0,
            'fl': 1.0,
            'x': max(self.params.x_in, 0.3),
            'y': max(0.2, 1 - self.params.x_in - 0.05),  # 估计值
            'w': max(self.params.w_in, 0.05),
            'rhob': self.params.rhob_in,
            'p': self.params.p_in,
            'z': self.params.HH - self.params.H0
        }
    
    def normalize_y(self, Y_physical):
        """物理量 → 归一化量"""
        norms = self.norms
        Y_norm = Y_physical.copy()
        for i, key in enumerate(['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']):
            Y_norm[i] /= norms[key]
        return Y_norm
    
    def denormalize_y(self, Y_norm):
        """归一化量 → 物理量"""
        norms = self.norms
        Y_physical = Y_norm.copy()
        for i, key in enumerate(['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']):
            Y_physical[i] *= norms[key]
        return Y_physical
    
    def normalized_bvp(self, z, Y_norm):
        """归一化变量的BVP函数"""
        # 转换为物理量
        Y_physical = self.denormalize_y(Y_norm)
        
        # 计算物理导数
        dY_physical = super().blast_furnace_bvp(z, Y_physical)
        
        # 转换为归一化导数
        norms = self.norms
        dY_norm = dY_physical.copy()
        for i, key in enumerate(['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']):
            dY_norm[i] *= norms['z'] / norms[key]
        
        return dY_norm
    
    def normalized_bc(self, ya_norm, yb_norm):
        """归一化变量的边界条件"""
        # 转换为物理量
        ya = self.denormalize_y(ya_norm)
        yb = self.denormalize_y(yb_norm)
        
        # 计算物理边界条件
        bc_physical = super().bc(ya, yb)
        
        # 归一化边界条件残差
        norms = self.norms
        bc_norm = bc_physical.copy()
        for i, key in enumerate(['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']):
            bc_norm[i] /= norms[key]
        
        return bc_norm
    
    def solve_normalized(self, save_csv=True, output_csv=None):
        """使用归一化变量求解"""
        z_guess, state = self._build_initial_guess()
        y_guess_physical = np.array([state[k] for k in ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']])
        y_guess_norm = self.normalize_y(y_guess_physical)

        H0_norm = self.params.H0 / self.norms['z']
        HH_norm = self.params.HH / self.norms['z']

        sol_norm, history = self._solve_with_decreasing_tol(
            self.normalized_bvp, self.normalized_bc, [H0_norm, HH_norm], y_guess_norm,
            tol_levels=[1e-1, 1e-2, 1e-3]
        )

        self._log_history(history)

        if sol_norm.success:
            z_physical = sol_norm.x * self.norms['z']
            y_physical = self.denormalize_y(sol_norm.y)
            self.z_solution = z_physical
            self.y_solution = y_physical

            plt.figure(figsize=(12, 8))
            variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.plot(z_physical, y_physical[i])
                plt.ylabel(variables[i])
                plt.xlabel('z (m)')
            plt.tight_layout()
            plt.close()

            if save_csv:
                if output_csv is None:
                    output_csv = f'sharp_R2_1200_1e-3_normalized.csv'
                df = pd.DataFrame(np.vstack((z_physical, y_physical)).T, columns=['z'] + variables)
                df.to_csv(output_csv, index=False)

            self.results = {
                "case_name": self.params.case_name,
                "H0": z_physical[0],
                "HH": z_physical[-1],
                "T_out": y_physical[0,0],
                "t_out": y_physical[1,-1],
                "fs_out": y_physical[2,-1],
                "fl_out": y_physical[3,-1],
                "x_out": y_physical[4,0],
                "y_out": y_physical[5,0],
                "w_out": y_physical[6,0],    
                "rhob_out": y_physical[7,-1],    
                "p_bottom": y_physical[8,-1]
            }

            return self.results
        else:
            raise RuntimeError("求解失败")
        
    def init_test(self):
        """使用归一化变量求解"""

        ## 读取CSV文件作为初值
        df = pd.read_csv('R2_1200_1e-3_normalized.csv')
        # 生成均匀分布的索引
        indices = np.linspace(0, len(df)-1, len(df), dtype=int)    # 测试初值的残差
        # 按索引取行
        sampled_df = df.iloc[indices]
        z_guess_physical = sampled_df['z'].values
        T = sampled_df['T'].values
        t = sampled_df['t'].values
        fs = sampled_df['fs'].values
        fl = sampled_df['fl'].values   
        x = sampled_df['x'].values
        y = sampled_df['y'].values
        w = sampled_df['w'].values
        rhob = sampled_df['rhob'].values
        p = sampled_df['p'].values

        y_guess_physical = np.array([T, t, fs, fl, x, y, w, rhob, p])

        # # 归一化初始猜测
        # z_guess_norm = z_guess_physical / self.norms['z']
        # y_guess_norm = self.normalize_y(y_guess_physical)
        
        # # 使用归一化的BVP求解
        # sol_norm = solve_bvp(
        #     self.normalized_bvp, 
        #     self.normalized_bc, 
        #     z_guess_norm, 
        #     y_guess_norm,
        #     tol = 1e-3,
        #     max_nodes = 10000,
        #     verbose = 2
        # )

        # # 反归一化结果
        # if sol_norm.success:
        #     z_physical = sol_norm.x * self.norms['z']
        #     y_physical = self.denormalize_y(sol_norm.y)
        #     # 绘制结果
        #     plt.figure(figsize=(12, 8))
        #     variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
        #     for i in range(9):
        #         plt.subplot(3, 3, i+1)
        #         plt.plot(z_physical, y_physical[i], label='solution')
        #         plt.ylabel(variables[i])
        #         plt.xlabel('z (m)')
        #     plt.tight_layout()
        #     plt.show()

        #     # 保存结果
        #     df = pd.DataFrame(np.vstack((z_physical, y_physical)).T, columns=['z'] + variables)
        #     # df.to_csv(f'{self.params.case_name}_{self.params.H0:.1f}-{self.params.HH:.1f}m_5e-6_normalized.csv', index=False)

        #     self.results = {
        #         "case_name": self.params.case_name,
        #         "H0": z_physical[0],
        #         "HH": z_physical[-1],
        #         "T_out": y_physical[0,0],
        #         "t_out": y_physical[1,-1],
        #         "fs_out": y_physical[2,-1],
        #         "fl_out": y_physical[3,-1],
        #         "x_out": y_physical[4,0],
        #         "y_out": y_physical[5,0],
        #         "w_out": y_physical[6,0],    
        #         "rhob_out": y_physical[7,-1],    
        #         "p_bottom": y_physical[8,-1]
        #     }

        #     return self.results
        # else:
        #     raise RuntimeError("求解失败")

        # ### 存yp_ex
        # from save_load import load_parameters
        # params = load_parameters("my_design")
        # model = NormalizedFurnaceModel(params)
        # variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
        # yp_ex = [model.dTdz(z_guess_physical,T,t,fs,fl,x,y,w,p),
        #          model.dtdz(z_guess_physical,T,t,fs,fl,x,y,w,p,rhob),
        #          model.dfsdz(z_guess_physical,T,t,fs,x,y,w,p),
        #          model.dfldz(z_guess_physical,T,t,fl,x,y,w,p),
        #          model.dxdz(z_guess_physical,T,t,fs,fl,x,y,w,p),
        #          model.dydz(z_guess_physical,T,t,fs,fl,x,y,w,p),
        #          model.dwdz(z_guess_physical,T,t,fs,fl,x,y,w,p),
        #          model.drhobdz(z_guess_physical,T,t,fs,fl,x,y,w,p),
        #          model.dpdz(z_guess_physical,T,x,y,w,p)]
        # df = pd.DataFrame(np.vstack((z_guess_physical, yp_ex)).T, columns=['z'] + variables)
        # df.to_csv(f'{self.params.case_name}_{self.params.H0:.1f}-{self.params.HH:.1f}m_1e-5_yp_ex_raw.csv', index=False)
        # ###

        ## 原始物理方程 ###
        sol = solve_bvp(
            self.blast_furnace_bvp, 
            self.bc, 
            z_guess_physical, 
            y_guess_physical,
            tol = 1e-3,
            max_nodes = 10000,
            verbose = 2
        )

        final_sol = solve_bvp(
            self.blast_furnace_bvp, 
            self.bc, 
            sol.x, 
            sol.y,
            tol = 1e-4,
            max_nodes = 10000,
            verbose = 2
        )

        # # 逐轮缩小容差求解
        # final_sol, history = self.solve_with_decreasing_tol(
        #     self.blast_furnace_bvp, 
        #     self.bc, 
        #     z_guess_physical, 
        #     y_guess_physical,
        #     tol_levels=[1e-3, 1e-4, 1e-5, 5e-6]
        # )
        # # 输出结果
        # print("\n=== 迭代历史 ===")
        # for i, record in enumerate(history):
        #     print(f"轮次 {i+1}: 容差={record['tol']:.1e}, "
        #         f"节点数={record['n_nodes']}, 成功={record['success']}")

        # 反归一化结果
        if final_sol.success:
            z_physical = final_sol.x
            y_physical = final_sol.y
            yp = final_sol.yp
            # 绘制结果
            plt.figure(figsize=(12, 8))
            variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.plot(z_physical, y_physical[i], label='solution')
                plt.ylabel(variables[i])
                plt.xlabel('z (m)')
            plt.tight_layout()
            plt.show()

            # 保存结果
            df = pd.DataFrame(np.vstack((z_physical, y_physical)).T, columns=['z'] + variables)
            # df.to_csv('sharp_R2_1200_1e-3_raw.csv', index=False)
            # df = pd.DataFrame(np.vstack((z_physical, yp)).T, columns=['z'] + variables)
            # df.to_csv(f'{self.params.case_name}_{self.params.H0:.1f}-{self.params.HH:.1f}m_1e-5_yp_raw.csv', index=False)

            self.results = {
                "case_name": self.params.case_name,
                "H0": z_physical[0],
                "HH": z_physical[-1],
                "T_out": y_physical[0,0],
                "t_out": y_physical[1,-1],
                "fs_out": y_physical[2,-1],
                "fl_out": y_physical[3,-1],
                "x_out": y_physical[4,0],
                "y_out": y_physical[5,0],
                "w_out": y_physical[6,0],    
                "rhob_out": y_physical[7,-1],    
                "p_bottom": y_physical[8,-1]
            }

            return self.results
        else:
            raise RuntimeError("求解失败")
        

class HCFurnaceModel(FurnaceModel):
    """
    高炉热量流模型
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def _relax_solver(self, z, state, update_fn, fields, tol=1e-3, max_iter=100, omega=0.5):
        """通用收敛求解器"""
        for _ in range(max_iter):
            new_state = update_fn(z, state)
            error = sum(norm(new_state[k] - state[k]) / norm(state[k]) for k in fields) / len(fields)
            if error < tol:
                return new_state
            for k in fields:
                state[k] = omega * new_state[k] + (1 - omega) * state[k]
        return state

    # Heat Current Method
    def Tt_hc(self,z,state):
        """[T,t,fs,fl,x,y,w,p,rhob]->[T_new,t_new]

        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, fl, x, y, w, p, rhob (numpy.ndarray): state variables.

        Returns:
            T_new (numpy.ndarray): temperature profile of gas. [K]
            t_new (numpy.ndarray): temperature profile of coke-bed. [K]
        """
        Dz = self.params.Diameter_BF(z)
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]


        def update_fn(z, current_state):
            current_state = self._clamp_state(**current_state)
            T, t, fs, fl, x, y, w, p, rhob = current_state['T'], current_state['t'], current_state['fs'], current_state['fl'], current_state['x'], current_state['y'], current_state['w'], current_state['p'], current_state['rhob']
            
            F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
            rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
            miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
            C,dCdT = self.HeatCapacity_Gas(T,x,y,w) # C (float): heat capacity of gas. [kcal / kg * K] ; dCdT (float): differential of C with T. [kcal / kg * K**2]
            Cs,dCsdt = self.HeatCapacity_Solid(t) # Cs (float): specific heat of solid particles. [kcal / kg * K] ; dCsdt (float): specific heat of solid particles differential T. [kcal / kg * K**2]

            G = rho * F / Az # G (float): mass velocity of gas. [kg / m2 * hr]
            Re = self.params.d_p * G / miu
            k = 0.06 # k (float): thermal conductivity of gas. [kcal / m * hr * K]
            Pr = C * miu / k
            Nu = 2.0 + 0.60*Re**(1/2)*Pr**(1/3)
            h_p = Nu * k / self.params.d_p # h_p (float): particle-to-fluid heat transfer coefficient. [kcal / m2 * hr * K]

            KA = 6 * (1-self.params.epsilon) * h_p * Az  / self.params.phi_o / self.params.d_p  # KA (float): Heat transfer coefficient. [kcal / m * hr * K]

            R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
            R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
            R3 = self.ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
            R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
            R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
            R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]
            R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): CO + H2O = CO2 + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]        

            weight1 = smooth_heaviside(fs-0.111,k=200)
            weight2 = smooth_heaviside(fs-0.333,k=200)

            H1 = np.zeros_like(z)
            H5 = np.zeros_like(z)
            mask = (fs < 0.222)
            H1[mask] = (1-weight1[mask])*-7.88e3 * 1/9 + weight1[mask]*7.12e3 * 2/9
            H5[mask] = (1-weight1[mask])*-2.8e3 * 1/9 + weight1[mask]*16.1e3 * 2/9
            H1[~mask] = (1-weight2[~mask])*7.12e3 * 2/9 + weight2[~mask]*-5.45e3 * 2/3
            H5[~mask] = (1-weight2[~mask])*16.1e3 * 2/9 + weight2[~mask]*6.5e3 * 2/3
            
            H2 = 40.8e3 # [kcal / kmol CO2]
            H3 = 31.13e3 # [kcal / kmol CO]
            H4 = 42.5e3 # [kcal / kmol CO2]
            H6 = 31.5e3 # [kcal / kmol CO]
            H7 = -9.84e3 # [kcal / kmol CO2]

            # t<1673K
            q2 = np.where(t<1200, (1.2507*0 + 0.7261*1)*R1 + 0.5246*R2 + 1.9768*R4 + 0.7143*R5 + 0.5364*R6,
                        (1.2507*0 + 0.7261*1)*R1 + 0.5246*(R2+R1+R4+R7) + 1.9768*R4 + 0.7143*R5 + 0.5364*R6)
            q4 = np.where(t<1200, -H1*R1 -H2*R2 -H4*R4 -H5*R5 -H6*R6 -H7*R7,
                        -H1*R1 -H2*(R2+R1+R4+R7) -H4*R4 -H5*R5 -H6*R6 -H7*R7)
            q5 = np.where(t<1200, 16*R1 + 12*R2 + 44*R4 + 16*R5 + 12*R6,
                        16*R1 + 12*(R2+R1+R4+R7) + 44*R4 + 16*R5 + 12*R6)

            G1 = rhob * self.params.Fs * (Cs + t*dCsdt) # solid   [kcal / hr * K]
            G2 = rho * F * (C + T*dCdT)    # gas     [kcal / hr * K]
            Q1 = Az*q4 + Az*Cs*t*q5                 # solid
            Q2 = 22.4*Az*C*q2*T + pai*Dz*self.params.U*(T-self.params.T_we) # gas       [kcal / m * hr]

            bkp = min(len(z[t<1200]), len(z)-2)
            Q1_bkp = Q1[bkp]
            Q2_bkp = Q2[bkp]

            G1 = (G1[1:] + G1[:-1]) / 2 
            G2 = (G2[1:] + G2[:-1]) / 2
            KA = (KA[1:] + KA[:-1]) / 2
            Q1 = (Q1[1:] + Q1[:-1]) / 2
            Q2 = (Q2[1:] + Q2[:-1]) / 2
            Q1[bkp] = Q1_bkp
            Q2[bkp] = Q2_bkp

            z_diff = np.diff(z)
            N = len(z_diff)
            tin = self.params.t_in
            Tin = self.params.T_in
            A_tT,a_tT = setAa_n(N, z_diff, KA, G1, G2, Tin, tin, Q1, Q2)
            X_tT = solve(A_tT, a_tT)
            t_new = X_tT[0:N+1].ravel()
            T_new = X_tT[(N+1):(2*N+2)].ravel()
            count = 0
            limit = 100
            while(norm(T_new-T)/norm(T) >= 1e-3 or norm(t_new-t)/norm(t) >= 1e-3) and (count < limit):
                count += 1
                t, T = t_new, T_new
                A_tT,a_tT = setAa_n(N, z_diff, KA, G1, G2, tin, Tin, Q1, Q2)
                X_tT = solve(A_tT, a_tT)
                t_new = X_tT[0:N+1].ravel()
                T_new = X_tT[(N+1):(2*N+2)].ravel()

            current_state.update({'T': T_new, 't': t_new})
            return current_state

        relaxed_state = self._relax_solver(z, state, update_fn, fields=['T', 't'])
        return relaxed_state['T'], relaxed_state['t']

    def xy_hc(self,z,state):
        """
        
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, fl, x, y, w, p (numpy.ndarray)
        Returns:
            x_new (numpy.ndarray): profile of molar fraction of CO in bulk of gas. [-]
            y_new (numpy.ndarray): profile of molar fraction of CO2 in bulk of gas. [-]
        """
        def update_fn(z, state):
            state = self._clamp_state(**state)
            T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['p']
            Dz = self.params.Diameter_BF(z) # Dz (float): Diameter of blast furnace. [m]
            Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]
            miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
            F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
            u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
            D_CO = self.DiffusionCoefficient_CO(t,p)
            rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
            Re = self.params.d_o * u * rho / miu

            Sc = miu / rho / D_CO
            Sh = 2.0 + 0.55*Re**(1/2)*Sc**(1/3)
            kf = self.TransferCoefficient_Gas(Sh,D_CO,self.params.d_o)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]

            epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
            xi = 0.238 * self.params.epsilon_o + 0.04
            Ds = D_CO * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of CO in reduced iron phase. [m2 / hr]

            k1 = 347 * np.exp(-3460/t) # k (float): rate constant of reaction. [m / hr]

            K1 = self.smooth_R1(t,fs) # K (float): equilibrium constant of reaction. [-]
            kappa_1 = pai * self.params.d_o**2 * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 / 22.4 / t / (1/kf + self.params.d_o/2*((1-fs+eps)**(-1/3) - 1)/Ds + ((1-fs+eps)**(2/3)*k1*(1+1/K1))**(-1))
            KA = Az*kappa_1 # transfer coefficient [Nm3 / m * hr]

            R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p)
            R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p)
            R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p)
            R6 = self.ReactionRate_6(z,T,t,x,y,w,p)
            R7 = self.ReactionRate_7(T,x,y,w,p)

            G1 = F/22.4 * (1+K1)/K1  # G1 (float): capacity flow of x. [kmol / hr]
            G2 = F/22.4 * (1+K1)/K1  # G2 (float): capacity flow of y. [kmol / hr]

            R2 = np.where(t<1200, R2, R2+R1+R4+R7)
            Q1 = Az*((K1-1)/(1+K1)*kappa_1*y + (x-2)*R2 + x*R4 + (x-1)*R6 + R7) * (1+K1)/K1
            Q2 = Az*(-(K1-1)/(1+K1)*kappa_1*y + (y+1)*R2 + (y-1)*R4 + y*R6 - R7) * (1+K1)/K1

            bkp = min(len(z[t<1200]), len(z)-2)
            Q1_bkp = Q1[bkp]
            Q2_bkp = Q2[bkp]

            G1 = (G1[1:] + G1[:-1]) / 2 
            G2 = (G2[1:] + G2[:-1]) / 2
            KA = (KA[1:] + KA[:-1]) / 2
            Q1 = (Q1[1:] + Q1[:-1]) / 2
            Q2 = (Q2[1:] + Q2[:-1]) / 2
            Q1[bkp] = Q1_bkp
            Q2[bkp] = Q2_bkp

            z_diff = np.diff(z)
            N = len(z_diff)
            xin = self.params.x_in
            yin = self.params.y_in
     
            A_xy,a_xy = setAa_s(N, z_diff, KA, G1, G2, xin, yin, Q1, Q2)
            X = solve(A_xy, a_xy)
            x_new = X[0:N+1].ravel()
            y_new = X[(N+1):(2*N+2)].ravel()
            count = 0
            limit = 100
            while(norm(x_new-x)/norm(x) >= 1e-3*N**0.5 or norm(y_new-y)/norm(y) >= 1e-3*N**0.5) and (count < limit):
                count += 1
                # print("xy_hc, count = ", count)
                x, y = x_new, y_new
                A_xy,a_xy = setAa_s(N, z_diff, KA, G1, G2, xin, yin, Q1, Q2)
                X = solve(A_xy, a_xy)
                x_new = X[0:N+1].ravel()
                y_new = X[(N+1):(2*N+2)].ravel()
     
            state.update({'x': x_new, 'y': y_new})
            return state

        relaxed_state = self._relax_solver(z, state, update_fn, fields=['x', 'y'])
        return relaxed_state['x'], relaxed_state['y']

    def w_hc(self,z,state):
        """
        
        Args:
            state (dict): dictionary of state variables.
            T, t, fs, fl, x, y, w, p (numpy.ndarray)
        Returns:
            w_new (numpy.ndarray): profile of molar fraction of H2 in bulk of gas. [-]
        """
        def update_fn(z, state):
            state = self._clamp_state(**state)
            T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['p']
            Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
            Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2] 
            F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
            u = F/Az * T/T_std * P_std/p # u (float): superficial velocity of gas. [m / hr]
            miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
            rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
            Re = self.params.d_o * u * rho / miu
            D_H2 = 3.960E-6*t**1.78 / (p/P_std) # D_H2 (float): diffusion coefficient of H2 in blast furnace gas. [m2 / hr]
            Sc = miu / rho / D_H2
            Sh = 2.0 + 0.55*Re**(1/2)*Sc**(1/3)
            kf = self.TransferCoefficient_Gas(Sh,D_H2,self.params.d_o)  # kf (float): gas-film mass transfer coefficient in reaction. [m / hr]
            epsilon_v = 0.53 + 0.47 * self.params.epsilon_o
            xi = 0.238 * self.params.epsilon_o + 0.04
            Ds = D_H2 * epsilon_v * xi # Ds (float): intraparticle diffusion coefficient of H2 in reduced iron phase. [m2 / hr]
            k,K = self.smooth_R5(t)  # smoothed k,K

            kappa_5 = pai * self.params.d_o**(2) * self.params.phi_o**(-1) * self.params.N_o * (p/P_std) * 273 / 22.4 / t / (1/kf + self.params.d_o/2*((1-fs+eps)**(-1/3) - 1)/Ds + ((1-fs+eps)**(2/3)*k*(1+1/K))**(-1))

            R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p)
            R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p)
            R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p)
            R6 = self.ReactionRate_6(z,T,t,x,y,w,p)
            R7 = self.ReactionRate_7(T,x,y,w,p)

            R2 = np.where(t<1200,R2,R2+R1+R4+R7)
            a_list = 22.4*Az*kappa_5/F
            b_list = 22.4*Az*(w*R2 + w*R4 - kappa_5*self.params.F_0*(self.params.w_0+self.params.v_0)/F/(1+K) + (w-1)*R6 - R7) / F

            R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p)
            a_list[R5<=0] = 0
            b_list[R5<=0] = 22.4*Az[R5<=0]*(w[R5<=0]*R2[R5<=0] + w[R5<=0]*R4[R5<=0] + (w[R5<=0]-1)*R6[R5<=0] - R7[R5<=0]) / F[R5<=0]
            
            bkp = min(len(z[t<1200]), len(z)-2)
            a_bkp = a_list[bkp]
            b_bkp = b_list[bkp]

            a_list = (a_list[1:] + a_list[:-1]) / 2
            b_list = (b_list[1:] + b_list[:-1]) / 2
            a_list[bkp] = a_bkp
            b_list[bkp] = b_bkp

            z_diff = np.diff(z)
            N = len(z_diff)
            win = self.params.w_in
            A_w,a_w = setAa_linear_n(N, z_diff, win, a_list, b_list)
            X_low = solve(A_w, a_w)
            w_new = X_low.ravel()
            count = 0
            limit = 100
            while(norm(w_new-w) >= 1e-4*N**0.5) and (count < limit):
                count += 1
                w = w_new
                A_w,a_w = setAa_linear_n(N, z_diff, win, a_list, b_list)
                X_low = solve(A_w, a_w)
                w_new = X_low.ravel()

            state.update({'w': w_new})
            return state

        relaxed_state = self._relax_solver(z, state, update_fn, fields=['w'])
        return relaxed_state['w']

    def p_hc(self,z,state):
        """
        
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, x, y, w, p (numpy.ndarray)
        Returns:
            p_new (numpy.ndarray): profile of pressure of gas. [Kg / m2]
        """
        p2_in = self.params.p_in**2

        state = self._clamp_state(**state)
        T, x, y, w = state['T'], state['x'], state['y'], state['w']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        F = self.VolumeRate_Gas(x,y)   # F (float): volume rate of flow of gas. [Nm3 / hr]
        rho = self.Density_Gas(x,y,w) # rho (float): density of blast furnace gas. [kg / Nm3]
        G = F * rho / (Az * self.params.epsilon) # G (float): mass velocity of gas. [kg / m2 * hr]
        miu = self.Viscosity_Gas(T) # miu (float): viscosity of blast furnace gas. [kg / m * hr]
        Re = self.params.d_p * G / miu
        fk = (1.75 + 150 * (1 - self.params.epsilon)) / Re

        a_list = fk * (1 - self.params.epsilon) * G**2 * P_std * T / (g_c * self.params.epsilon**3 * self.params.d_p * rho * T_std)
        
        a_list = (a_list[1:] + a_list[:-1]) / 2

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp,a_temp = setAa_p(N, z_diff, p2_in, a_list)
        X_temp = solve(A_temp, a_temp)
        # print(X_temp.shape)
        p2_new = X_temp.ravel()
        p_new = np.sqrt(p2_new)

        return p_new

    def fs_hc(self,z,state):
        """
        
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, x, y, w, p (numpy.ndarray)
        Returns:
            fs_new (numpy.ndarray): profile of fraction of reduction of iron ore. [-]
        """
        fs_in = self.params.fs_in

        def update_fn(z, state):
            state = self._clamp_state(**state)
            T, t, fs, x, y, w, p = state['T'], state['t'], state['fs'], state['x'], state['y'], state['w'], state['p']
            Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
            Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

            R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p)
            # R3 = ReactionRate_3(t,fs)
            R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p)

            dd = Az * (R1 + R5) / 3 / self.params.Fs / self.params.c_H0

            a_list = dd
            a_list = (a_list[1:] + a_list[:-1]) / 2

            z_diff = np.diff(z)
            N = len(z_diff)
            A_temp,a_temp = setAa_constant_s(N, z_diff, fs_in, a_list)
            X_temp = solve(A_temp, a_temp)
  
            fs_new = X_temp.ravel()
            state.update({'fs': fs_new})
            return state
        relaxed_state = self._relax_solver(z, state, update_fn, fields=['fs'])
        return relaxed_state['fs']


    def fl_hc(self,z,state):
        """
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fl, x, y, w, p (numpy.ndarray)
        Returns:
            fl_new (numpy.ndarray): profile of fraction of decomposition of limestone. [-]
        """
        fl_in = self.params.fl_in
        def update_fn(z, state):
            state = self._clamp_state(**state)
            T, t, fl, x, y, w, p = state['T'], state['t'], state['fl'], state['x'], state['y'], state['w'], state['p']
            Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
            Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

            R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p)

            a_list = Az*R4 / self.params.Fs / self.params.c_L0
            a_list = (a_list[1:] + a_list[:-1]) / 2

            z_diff = np.diff(z)
            N = len(z_diff)
            A_temp,a_temp = setAa_constant_s(N, z_diff, fl_in, a_list)
            X_temp = solve(A_temp, a_temp)

            fl_new = X_temp.ravel()
            state.update({'fl': fl_new})
            return state
        relaxed_state = self._relax_solver(z, state, update_fn, fields=['fl'])
        return relaxed_state['fl']


    def rhob_hc(self,z,state):
        """
        Args:
            z (numpy.ndarray): axial position of coke-bed. [m]
            T, t, fs, fl, x, y, w, p, rhob (numpy.ndarray)
        Returns:
            rhob_new (numpy.ndarray): profile of . [kg / m3]
        """
        rhob_in = self.params.rhob_in

        state = self._clamp_state(**state)
        T, t, fs, fl, x, y, w, p = state['T'], state['t'], state['fs'], state['fl'], state['x'], state['y'], state['w'], state['p']

        Dz = self.params.Diameter_BF(z) # Dz (float): diameter of coke-bed. [m]
        Az = pai * (Dz/2)**2 # Az (float): cross-sectional area of coke-bed. [m2]

        R1 = self.ReactionRate_1(z,T,t,fs,x,y,w,p) # R1 (float): 1/3 Fe2O3 + CO = 2/3 Fe + CO2 reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R2 = self.ReactionRate_2(z,T,t,fs,x,y,w,p) # R2 (float): C + CO2 = 2CO reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        # R3 = ReactionRate_3(t,fs) # R3 (float): FeO(l) + C(s) = Fe(l) + CO(g) reaction rate per unit volume of bed. [kmol CO / m3 bed * hr]
        R4 = self.ReactionRate_4(z,T,t,fl,x,y,w,p) # R4 (float): CaCO3 = CaO + CO2 reaction rate per unit volume of bed. [kmol CO2 / m3 bed * hr]
        R5 = self.ReactionRate_5(z,T,t,fs,x,y,w,p) # R5 (float): 1/3 Fe2O3 + H2 = 2/3 Fe + H2O reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]
        R6 = self.ReactionRate_6(z,T,t,x,y,w,p) # R6 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2O / m3 bed * hr]
        R7 = self.ReactionRate_7(T,x,y,w,p) # R7 (float): C + H2O = CO + H2 reaction rate per unit volume of bed. [kmol H2 / m3 bed * hr]

        R2 = np.where(t<1200, R2, R2 + R1+R4+R7)

        dd = -Az * ((16+12*0)*R1 + 12*R2 + 44*R4 + 16*R5 + 12*R6) / self.params.Fs

        a_list = dd
        bkp = min(len(z[t<1200]), len(z)-2)
        a_bkp = a_list[bkp]
        a_list = (a_list[1:] + a_list[:-1]) / 2
        a_list[bkp] = a_bkp

        z_diff = np.diff(z)
        N = len(z_diff)
        A_temp,a_temp = setAa_constant_s(N, z_diff, rhob_in, a_list)
        X_temp = solve(A_temp, a_temp)
        # print(X_temp.shape)
        rhob_new = X_temp.ravel()

        return rhob_new
    
