# parameters.py
import numpy as np

from constant import pai

class FurnaceParameters:
    """高炉参数类"""
    
    def __init__(self, case_name="default_case"):
        self.case_name = case_name
        
        # 几何参数 Design parameters
        self.D0 = 6700e-3 # [m] diameter of stockline
        self.D1 = 9000e-3 # [m] diameter of hearth
        self.Db = 10000e-3 # [m] diameter of bosh
        self.Ls = 18700e-3 # [m] height of shaft
        self.La = 2500e-3 # [m] height of bosh
        self.Lb = 3500e-3 # [m] height between bosh and tuyere
        
        # 操作参数 Operation parameters
        self.epsilon = 0.22 # [-] fractional void in bed
        self.p0 = 1.433e4 # [Kg / m2] top pressure
        self.d_p = 0.01935 # [m] diameter of solid particles, for h_p calculation
        self.F_b = 2942 # volume rate of dry blast [Nm3 / min]
        self.F_0 = 2.65e5 # volume rate of flow of top gas. [Nm3 / hr]
        self.w_0 = 0.041 # molar fraction of hydrogen of top gas. [-]
        self.v_0 = 0.027 # molar fraction of water vapor of top gas. [-]
        self.c_H0 = 4.505 # initial concentration of hematite. [kmol / m3 bed]
        self.c_L0 = 0.4281 # initial concentration of limestone. [kmol / m3 bed]
        self.T_we = 35 + 273 # exit tempereture of cooling water. [K]
        self.U = 10 # [kcal / m2 * hr * K] estimated value of overall heat transfer coefficient based on inner surface area of furnace-wall.
        self.W_o = 230.63e3 # mass rate of flow of iron ore. [kg(ore) / hr]
        self.W_c = 68.0e3 # mass rate of flow of coke. [kg(coke) / hr]
        self.W_L = 11.226e3 # mass rate of flow of limestone. [kg(limestone) / hr]

        # 材料参数 Material parameters
        self.d_o = 0.02 # [m] diameter of ore
        self.phi_o = 0.8 # [-] shape factor of ore
        self.epsilon_o = 0.40 # [-] porosity of ore
        self.rho_po = 1948 # [kg/m3] apparent density of solid particles of ore
        self.d_c = 0.0530 # [m] diameter of coke
        self.phi_c = 0.7 # [-] shape factor of coke
        self.epsilon_c = 0.45 # [-] porosity of coke
        self.rho_pc = 477 # [kg/m3] apparent density of solid particles of coke
        self.d_L = 0.0331 # [m] diameter of limestone
        self.phi_L = 0.9 # [-] shape factor of limestone
        self.epsilon_L = 0.25 # [-] porosity of limestone
        self.rho_pL = 1599 # [kg/m3] apparent density of solid particles of limestone
        
        # Calculated parameters
        self.F_o = self.W_o / self.rho_po # [m3 bed / hr] volume rate of ore
        self.F_c = self.W_c / self.rho_pc # [m3 bed / hr] volume rate of coke
        self.F_L = self.W_L / self.rho_pL # [m3 bed / hr] volume rate of limestone
        self.Fs = self.F_o + self.F_c + self.F_L # volume rate of solid particles. [m3 bed / hr]
        self.N_o = (1-self.epsilon) / (4/3*pai*(self.d_o/2)**3) * self.F_o/self.Fs # [1/m3 bed] number of particles per unit volume of bed
        self.N_c = (1-self.epsilon) / (4/3*pai*(self.d_c/2)**3) * self.F_c/self.Fs # [1/m3 bed] number of particles per unit volume of bed
        self.N_L = (1-self.epsilon) / (4/3*pai*(self.d_L/2)**3) * self.F_L/self.Fs # [1/m3 bed] number of particles per unit volume of bed
        
        self.F_mH = self.c_H0 * self.Fs # molar rate of flow of hematite. [kmol / hr]
        self.rho_S = 2600 # density of slag. [kg / m3]
        self.W_S = 49e3 # mass rate of flow of slag. [kg / hr] (calculated by Muchi 1967)
        self.rho_W = 3800 # density of wustite. [kg / m3]

        # 边界条件 Boundary conditions
        self.T_in = 1672 # [K] inlet temperature of gas
        self.t_in = 505 # [K] inlet temperature of solid
        self.fs_in = 0.0 # [-] inlet reduction faction of ore
        self.fl_in = 0.0 # [-] inlet decomposition faction of limestone
        self.x_in = 0.365 # [-] inlet mole fraction of CO
        self.y_in = 0.0 # [-] inlet mole fraction of CO2
        self.w_in = 0.063 # [-] inlet mole fraction of H2
        self.rhob_in = 1268 # [kg / m3 bed] inlet density of bed
        self.p_in = self.p0 # [Kg / m2] top pressure

        # 初始节点 Initial nodes
        self.H0 = 0.0 # [m] height of the starting point of calculation
        self.H1 = 4.00
        self.H2 = 12.00
        self.H3 = 16.00
        self.HH = 20 # [m] height of the end point of calculation
        # 0 m
        self.value0 = [595, 505, 0, 0, 0.243, 0.174, 0.04, 1268, 1.48e4]  # 
        # 4 m
        self.value1 = [830, 833, 0.04, 0, 0.273, 0.149, 0.04, 1259, 1.94e4]  #
        # 12 m
        self.value2 = [1100, 1090, 0.35, 0.17, 0.347, 0.077, 0.04, 1220, 2.49e4]  # 
        # 16 m
        self.value3 = [1189, 1189, 0.67, 0.94, 0.415, 0, 0.043, 1157, 2.69e4]  # 
        # 20 m
        self.valueH = [1672, 1548, 0.92, 1, 0.365, 0.0, 0.063, 1040, 2.90e4]  # 


        # 数值参数 Numerical parameters
        self.initial_mesh = 2000
        # self.time_steps = 1000
        # self.tolerance = 1e-6

    def Diameter_BF(self, z):
        """Diameter of blast furnace

        Args:
            z (float): height from the stock line. [m]
        
        Returns:
            D (float): Diameter of blast furnace. [m]
        """
        # z = z * 25.25/23 # 20251105

        z = np.asarray(z)
        D = np.zeros_like(z)
        mask1 = (z <= self.Ls)
        mask2 = (z > self.Ls) & (z <= self.Ls+self.La)
        mask3 = (z > self.Ls+self.La) & (z <= self.Ls+self.La+self.Lb)
        mask4 = (z > self.Ls+self.La+self.Lb)
        # D[mask1] = D0 + 2*z[mask1]/np.tan(omega_1)
        D[mask1] = self.D0 + z[mask1]*((self.Db-self.D0)/self.Ls)
        D[mask2] = self.Db
        D[mask3] = self.Db - (z[mask3]-self.Ls-self.La)/self.Lb*(self.Db-self.D1)
        D[mask4] = self.D1

        return D # [m2]
    


def create_standard_case(case_type="default"):
    """创建标准算例"""
    if case_type == "O2_rich_0.03":
        params = FurnaceParameters("O2_rich_0.03")

        params.W_o = 264e3 # mass rate of flow of iron ore. [kg(ore) / hr]
        params.W_c = 77.8e3 # mass rate of flow of coke. [kg(coke) / hr]
        params.W_L = 12.858e3 # mass rate of flow of limestone. [kg(limestone) / hr]

        # params.T_in = 1672 # [K] inlet temperature of gas
        # params.t_in = 505 # [K] inlet temperature of solid
        # params.fs_in = 0.0 # [-] inlet reduction faction of ore
        # params.fl_in = 0.0 # [-] inlet decomposition faction of limestone
        # params.x_in = 0.365 # [-] inlet mole fraction of CO
        # params.y_in = 0.0 # [-] inlet mole fraction of CO2
        # params.w_in = 0.063 # [-] inlet mole fraction of H2
        # params.rhob_in = 1268 # [kg / m3 bed] inlet density of bed
        # # 0 m
        # params.value0 = [595 / T_scale, 505 / t_scale, 0 / fs_scale, 0 / fl_scale, 0.243 / x_scale, 0.174 / y_scale, 0.04 / w_scale, 1268 / rhob_scale, 1.48e4 / p_scale]  # 
        # # 4 m
        # params.value1 = [830 / T_scale, 833 / t_scale, 0.04 / fs_scale, 0 / fl_scale, 0.273 / x_scale, 0.149 / y_scale, 0.04 / w_scale, 1259 / rhob_scale, 1.94e4 / p_scale]  #
        # # 12 m
        # params.value2 = [1100 / T_scale, 1090 / t_scale, 0.35 / fs_scale, 0.17 / fl_scale, 0.347 / x_scale, 0.077 / y_scale, 0.04 / w_scale, 1220 / rhob_scale, 2.49e4 / p_scale]  # 
        # # 16 m
        # params.value3 = [1189 / T_scale, 1189 / t_scale, 0.67 / fs_scale, 0.94 / fl_scale, 0.415 / x_scale, 0.0 / y_scale, 0.043 / w_scale, 1157 / rhob_scale, 2.69e4 / p_scale]  # 
        # # 20 m
        # params.valueH = [1672 / T_scale, 1548 / t_scale, 0.92 / fs_scale, 1 / fl_scale, 0.365 / x_scale, 0.0 / y_scale, 0.063 / w_scale, 1040 / rhob_scale, 2.90e4 / p_scale]  #        
    
    elif case_type == "O2_rich_0.07":
        params = FurnaceParameters("O2_rich_0.07")

        params.W_o = 309e3 # mass rate of flow of iron ore. [kg(ore) / hr]
        params.W_c = 90.2e3 # mass rate of flow of coke. [kg(coke) / hr]
        params.W_L = 15.049e3 # mass rate of flow of limestone. [kg(limestone) / hr]

        # params.T_in = 1672 # [K] inlet temperature of gas
        # params.t_in = 505 # [K] inlet temperature of solid
        # params.fs_in = 0.0 # [-] inlet reduction faction of ore
        # params.fl_in = 0.0 # [-] inlet decomposition faction of limestone
        # params.x_in = 0.365 # [-] inlet mole fraction of CO
        # params.y_in = 0.0 # [-] inlet mole fraction of CO2
        # params.w_in = 0.063 # [-] inlet mole fraction of H2
        # params.rhob_in = 1268 # [kg / m3 bed] inlet density of bed
        # # 0 m
        # params.value0 = [595 / T_scale, 505 / t_scale, 0 / fs_scale, 0 / fl_scale, 0.243 / x_scale, 0.174 / y_scale, 0.04 / w_scale, 1268 / rhob_scale, 1.48e4 / p_scale]  # 
        # # 4 m
        # params.value1 = [830 / T_scale, 833 / t_scale, 0.04 / fs_scale, 0 / fl_scale, 0.273 / x_scale, 0.149 / y_scale, 0.04 / w_scale, 1259 / rhob_scale, 1.94e4 / p_scale]  #
        # # 12 m
        # params.value2 = [1100 / T_scale, 1090 / t_scale, 0.35 / fs_scale, 0.17 / fl_scale, 0.347 / x_scale, 0.077 / y_scale, 0.04 / w_scale, 1220 / rhob_scale, 2.49e4 / p_scale]  # 
        # # 16 m
        # params.value3 = [1189 / T_scale, 1189 / t_scale, 0.67 / fs_scale, 0.94 / fl_scale, 0.415 / x_scale, 0.0 / y_scale, 0.043 / w_scale, 1157 / rhob_scale, 2.69e4 / p_scale]  # 
        # # 20 m
        # params.valueH = [1672 / T_scale, 1548 / t_scale, 0.92 / fs_scale, 1 / fl_scale, 0.365 / x_scale, 0.0 / y_scale, 0.063 / w_scale, 1040 / rhob_scale, 2.90e4 / p_scale]  # 
    
    else:  # default
        params = FurnaceParameters("default_case")
    
    return params

def quick_modify(base_params, **changes):
    """快速修改参数"""
    new_params = FurnaceParameters()
    # 复制基础参数
    for key, value in base_params.__dict__.items():
        setattr(new_params, key, value)
    # 更新修改的参数
    for key, value in changes.items():
        if hasattr(new_params, key):
            setattr(new_params, key, value)
    return new_params