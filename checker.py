import numpy as np
from scipy.interpolate import interp1d

class Checker:
    def __init__(self, bridge):
        assert bridge._checking, '当前不在检算状态.'
        self.bridge = bridge



################### 检算 ############################
#                                                  #
#  警告：以下内容较繁琐且可读性较差，具体细节请参照规范！   #
#                                                  #
####################################################




############# 疲劳检算 ##############

    def get_gamma_rho(self, rho):
        '''应力比修正系数'''
        
        RHO         = [-float('inf'),  -4.5,  -4.0, -3.5, -3.0, -2.0, -1.8, -1.6, -1.4, -1.2, float('inf')]

        gamma_RHO   = [         0.21,  0.21,  0.23, 0.25, 0.28, 0.36, 0.38, 0.41, 0.43, 0.46, 0.46]

        gamma_rho = interp1d(np.array(RHO), np.array(gamma_RHO), 'linear')(rho)

        return gamma_rho            

    
    def get_gamma_n(self, unit):
        '''以受拉为主的构件的损伤修正系数'''
        unit_axial_forces = np.around(np.array(unit.axial_forces), decimals=3)
        unit_axial_forces_pos_indices = np.where(unit_axial_forces > 0)[0]
        
        start = unit_axial_forces_pos_indices[0] - 1
        end = unit_axial_forces_pos_indices[-1] + 1
        
        load_len = (end - start) * 0.1  # 影响线加载长度

        load_lens = [   0,    4,    5,    8,   16,   20,   30, float('inf')]

        gamma_N   = [1.50, 1.50, 1.45, 1.30, 1.20, 1.10, 1.00, 1.00]

        gamma_n = interp1d(np.array(load_lens), np.array(gamma_N), 'linear')(load_len)

        return gamma_n


    def fatigue_check_unit(self, unit):
        '''疲劳检算'''
        _, _, t2, _ = unit.beam_section_data

        gamma_d       = 1.16              
        gamma_t       = 1.0 if t2 <= 0.025 else (25 / (t2 * 10**3))**(1 / 4)
        gamma_n       = None
        gamma_n_prime = 1
        sigma_0       = 130700
        
        qualified = None
        surplus = None
        
        if unit.is_zero:   # 0杆
            #print('skip ', unit.num)
            qualified = True

        else:
            
            # 疲劳最大活载 N_kf
            pos_max, neg_min = unit.N_kf
            mu_f = 1 + (18 / (40 + self.bridge.length)) - 1
            h = self.bridge.h
            A_j = unit.A_j
            
            sigma_max = (pos_max * (1 + mu_f) + h) / A_j
            sigma_min = (neg_min * (1 + mu_f) + h) / A_j
            sigma_max, sigma_min = np.around(sigma_max, decimals=3), np.around(sigma_min, decimals=3)
            
            
            if sigma_max <= 0:
                qualified = True
            
            else:
                rho = sigma_min / sigma_max
                if rho >= -1:
                    gamma_n = self.get_gamma_n(unit)
                    if gamma_d * gamma_n * (sigma_max - sigma_min) < gamma_t * sigma_0:
                        qualified = True
                        surplus = (gamma_d * gamma_n * (sigma_max - sigma_min) / gamma_t - sigma_0) / sigma_0
                    else:
                        qualified = False

                else: # rho < -1
                    gamma_rho = self.get_gamma_rho(rho)
                    if gamma_d * gamma_n_prime * sigma_max <= gamma_t * gamma_rho * sigma_0:
                        qualified = True
                        surplus = (gamma_d * gamma_n_prime * sigma_max / (gamma_t * gamma_rho) - sigma_0) / sigma_0
                    else:
                        qualified = False

        
        unit.fatigue_qualified = qualified
        unit.fatigue_surplus = surplus

        return qualified
        
        
    def fatigue_check(self):
        qualities = []
        not_qualified_units_nums = []
        for unit in self.bridge.units.values():
            qualified = self.fatigue_check_unit(unit)
            qualities.append(qualified)

            if not qualified:
                not_qualified_units_nums.append(unit.num)

        return np.array(qualities).all()

        
    def show_fatigue_check_results(self, show_qualified=False):

        header = '\n单元\t合格\t富余率'
        title = '### 疲劳检算 ###'.center(len(header))
        print(title + header)

        for unit in self.bridge.units.values():
            if unit.fatigue_qualified and show_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '是')
            elif not unit.fatigue_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '否')

            if unit.surplus is not None:
                qualified_info += '%.3f' % (unit.surplus)
            else:
                qualified_info += '\\'

            print(qualified_info)

        print()

            
############# 强度检算 ##############            
     
    def strength_check_unit(self, unit):
        '''杆件单元强度检算'''

        sigma = 200 * 10**3  # KPa
        
        qualified = None
        surplus = None
        
        N = unit.N 
        A_j = unit.A_j

        if unit.num == 30:
            print(np.around(abs(N) / A_j, decimals=3))

        qualified = True if ((abs(N) / A_j) < sigma).all() else False
        surplus = (np.max(np.abs(N)) / A_j - sigma) / sigma
        
        unit.strength_qualified = qualified
        unit.strength_surplus = surplus
        
        return qualified
              
        

    def strength_check(self):
        qualities = []
        not_qualified_units_nums = []
        for unit in self.bridge.units.values():
            qualified = self.strength_check_unit(unit)
            qualities.append(qualified)

            if not qualified:
                not_qualified_units_nums.append(unit.num)

        return np.array(qualities).all()



    def show_strength_check_results(self, show_qualified=False):

        header = '\n单元\t合格\t富余率'
        title = '### 强度检算 ###'.center(len(header))
        print(title + header)

        for unit in self.bridge.units.values():
            if unit.strength_qualified and show_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '是')
            elif not unit.strength_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '否')

            if unit.surplus is not None:
                qualified_info += '%.3f' % (unit.surplus)
            else:
                qualified_info += '\\'

            print(qualified_info)

        print()

            
############### 刚度检算 ##################
    def stiffness_check_unit(self, unit):
        ''''''
        
        qualified = None
        
        lambda_x, lambda_y = unit.lambda_
        
        N = unit.N
        if min(N) >= 0 and unit.num not in (self.bridge.top_chord_units_nums + self.bridge.bottom_chord_nodes_nums):
            thresh = (0, 180)
        else:
            thresh = (0, 100)

        # if lambda_x < thresh and lambda_y < thresh:
        if thresh[0] < lambda_x < thresh[1] and thresh[0] < lambda_y < thresh[1]:
            qualified = True
        else:
            qualified = False
        
        unit.stiffness_qualified = qualified

        return qualified


    def stiffness_check(self):
        qualities = []
        not_qualified_units_nums = []
        for unit in self.bridge.units.values():
            qualified = self.stiffness_check_unit(unit)
            qualities.append(qualified)

            if not qualified:
                not_qualified_units_nums.append(unit.num)

        return np.array(qualities).all()
    

    def show_stiffness_check_results(self):

        header = '\n单元\t合格\t富余率'
        title = '### 刚度检算 ###'.center(len(header))
        print(title + header)

        for unit in self.bridge.units.values():
            if unit.stiffness_qualified and show_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '是')
            elif not unit.stiffness_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '否')

            if unit.surplus is not None:
                qualified_info += '%.3f' % (unit.surplus)
            else:
                qualified_info += '\\'

            print(qualified_info)

        print()


############### 整体稳定检算 ##################
    
    def get_phi_1x(self, lambda_x):

        lambda_X = [0, 30, 40, 50,  
                    60, 70, 80, 90, 
                    100, 110, 120, 130,
                    140, 150, float('inf')]

        phi_1X = [0.900, 0.900, 0.867, 0.804, 
                  0.733, 0.655, 0.583, 0.517,  
                  0.454, 0.396, 0.346, 0.298, 
                  0.254, 0.214, 0.214]

        phi_1x = interp1d(np.array(lambda_X), np.array(phi_1X), 'linear')(lambda_x)

        return phi_1x

    
    def get_phi_1y(self, lambda_y):

        lambda_Y = [0, 30, 40, 50,  
                    60, 70, 80, 90, 
                    100, 110, 120, 130,
                    140, 150, float('inf')]

        phi_1Y = [0.900, 0.900, 0.823, 0.747, 
                  0.677, 0.609, 0.544, 0.483,  
                  0.424, 0.371, 0.327, 0.287, 
                  0.249, 0.212, 0.212]

        phi_1y = interp1d(np.array(lambda_Y), np.array(phi_1Y), 'linear')(lambda_y)

        return phi_1y


    
    def overall_stability_check_unit(self, unit):
        '''整体稳定检算'''

        sigma = 200 * 10**3  # KPa
        
        qualified = None
        surplus = None
        
        N = unit.N
        A_m = unit.A_m

        lambda_x, lambda_y = unit.lambda_
        phi_1 = min(self.get_phi_1x(lambda_x), self.get_phi_1y(lambda_y))

        if min(N) > 0:
            qualified = True
        else:
            qualified = (abs(min(N)) / A_m) <= (phi_1 * sigma)

        surplus = (N / (A_m * phi_1) - sigma) / sigma

        unit.overall_stability_qualified = qualified
        unit.overall_stability_surplus = surplus

        return qualified


    
    def overall_stability_check(self):
        
        qualities = []
        not_qualified_units_nums = []
        for unit in self.bridge.units.values():
            qualified = self.overall_stability_check_unit(unit)
            qualities.append(qualified)

            if not qualified:
                not_qualified_units_nums.append(unit.num)

        return np.array(qualities).all()

    
    def show_overall_stability_check_results(self):

        header = '\n单元\t合格\t富余率'
        title = '### 整体稳定检算 ###'.center(len(header))
        print(title + header)

        for unit in self.bridge.units.values():
            if unit.overall_stability_qualified and show_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '是')
            elif not unit.overall_stability_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '否')

            if unit.surplus is not None:
                qualified_info += '%.3f' % (unit.surplus)
            else:
                qualified_info += '\\'

            print(qualified_info)

        print()
            

############### 局部稳定检算 ##################
    def local_stability_check_unit(self, unit):

        qualified = None
        surplus = None
        
        if unit.is_zero:
            qualified = True
        
        else:
            N = unit.N
            if min(N) > 0:
                qualified = True
            
            else:
                b2, t1, t2, b1 = unit.beam_section_data
                lambda_x, lambda_y = unit.lambda_

                # if unit.num == 20:
                #     print(b2 / t2, b1 / t1)

                x_qualified = y_qualified = None

                if lambda_x < 50:
                    x_qualified = True if (b2 / t2 <= 30) else False
                else:
                    x_qualified = True if (b2 / t2 <= 0.4 * lambda_x + 10) else False

                if lambda_y < 50:
                    y_qualified = True if (b1 / t1 <= 12) else False
                else:
                    y_qualified = True if (b1 / t1 <= 0.14 * lambda_y + 5) else False

        
                qualified = x_qualified and y_qualified

        unit.local_stability_qualified = qualified
        
        return qualified

    
    def local_stability_check(self):

        qualities = []
        not_qualified_units_nums = []
        for unit in self.bridge.units.values():
            qualified = self.local_stability_check_unit(unit)
            qualities.append(qualified)

            if not qualified:
                not_qualified_units_nums.append(unit.num)

        return np.array(qualities).all()

    
    def show_local_stability_check_results(self):

        header = '\n单元\t合格\t富余率'
        title = '### 局部稳定检算 ###'.center(len(header))
        print(title + header)

        for unit in self.bridge.units.values():
            if unit.local_stability_qualified and show_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '是')
            elif not unit.local_stability_qualified:
                qualified_info = '%2d\t%s\t' % (unit.num, '否')

            if unit.surplus is not None:
                qualified_info += '%.3f' % (unit.surplus)
            else:
                qualified_info += '\\'

            print(qualified_info)

        print()



################ 梁体挠度 ####################
    def deflection_check(self):
        max_vdisp = 0.
        for node in self.nodes.values():
            node_max_vdisp = np.max(np.abs(node.max_vdisps))
            max_vdisp = np.max([max_vdisp, node_max_vdisp])
        
        max_vdisp *= 10 ** -3
        if max_vdisp < self.length / 900:
            qualified = True
        else:
            qualified = False

        self.deflection_qualified = qualified



##########################################


    def check_all(self):
        self.stiffness_check()
        self.local_stability_check()
        self.fatigue_check()
        self.overall_stability_check()
        self.strength_check()

        self.show_stiffness_check_results()
        self.show_local_stability_check_results()
        self.show_fatigue_check_results()
        self.show_overall_stability_check_results()
        self.show_strength_check_results()



