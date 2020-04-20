

class Checker:
	def __init__(self, bridge):
		self.bridge = bridge


################### 检算 ############################
#                                                  #
#  警告：以下内容较繁琐且可读性较差，具体细节请参照规范！   #
#                                                  #
####################################################



############# 疲劳检算 ##############
    def inter(self, mapping, x3, x1, x2):
        # 线性插值
        #  x1 < x3 < x2
        y1, y2 = mapping(x1), mapping(x2)
        return y1 + (x3 - x1) / (x2 - x1) * (y2 - y1)

    
    def get_gamma_rho(self, rho):
        
        gamma_rho_map = {
            -4.5: 0.21, -4.0: 0.23, -3.5: 0.25, -3.0: 0.28,
            -2.0: 0.36, -1.8: 0.38, -1.6: 0.41, -1.4: -0.43,
            -1.2: 0.46, 
        }
        fixed_rho = [-4.5, -4.0, -3.5, -3.0, -2.0, -1.8, -1.6, -1.4, -1.2]
        
        if rho in gamma_rho_map.keys():
            gamma_rho = gamma_rho_map[rho]
        elif rho < -4.5:
            gamma_rho = gamma_rho_map[-4.5]
        elif rho > -1.2:
            gamma_rho = gamma_rho_map[-1.2]
        else:
            for i in range(len(fix_rho)):
                if rho < fixed_rho[i]:
                    gamma_rho = inter(rho, fixed_rho[i - 1], fixed_rho[i])
                    break
        return gamma_rho

    
    def get_gamma_n(self, unit):
        
        unit_axial_forces = np.around(np.array(unit.axial_forces), decimals=3)
        unit_axial_forces_pos_indices = np.where(unit_axial_forces > 0)[0]
        
        start = unit_axial_forces_pos_indices[0] - 1
        end = unit_axial_forces_pos_indices[-1] + 1
        
        fixed_load_lens = [30, 20, 16, 12, 8, 5, 4]
        load_len = (end - start) * 0.1  # 影响线加载长度
        
        gamma_n_mapping = {
            30: 1.00, 20: 1.10, 16: 1.20, 
             8: 1.30,  5: 1.45,  4: 1.50
        }
        if load_len in fixed_load_lens:
            gamma_n = gamma_n_mapping[load_len]

        elif load_len > 30:
            gamma_n =  gamma_n_mapping[30]

        elif load_len < 4:
            gamma_n = gamma_n_mapping[4]

        else:
            for i in range(len(fix_load_lens)):
                if load_len > fixed_load_lens[i]:
                    gamma_n = inter(load_len, fixed_load_lens[i], fixed_load_lens[i - 1])
                    break

        return gamma_n
    
    
    def fatigue_check(self, unit, params):
        '''疲劳检算'''
        _, _, t2, _ = unit.beam_section_data

        gamma_d       = 1              
        gamma_t       = 1.0 if t2 <= 0.025 else (25 / (t2 * 10**3))**(1 / 4)
        gamma_n       = None
        gamma_n_prime = 1
        sigma_0       = 130700
        
        qualified = None
        fatigue_surplus = None
        
        if unit.is_zero_bar:   # 0杆
            #print('skip ', unit.num)
            qualified = True

        else:
            unit_axial_forces = np.array(unit.axial_forces)
            # 最大活载 N_k
            pos_max, neg_min = np.around(unit.max_force, decimals=3)
            
            # 疲劳最大活载 N_k_f
            pos_max += (((unit_axial_forces < 0) * unit_axial_forces) * 6.4).sum()
            neg_min += (((unit_axial_forces > 0) * unit_axial_forces) * 6.4).sum()
            
            mu_f = 1 + (18 / (40 + self.length)) - 1

            A_j = self.A_j
            
            sigma_max = (pos_max * (1 + mu_f) + self.h) / A_j
            sigma_min = (neg_min * (1 + mu_f) + self.h) / A_j
            sigma_max, sigma_min = np.around(sigma_max, decimals=3), np.around(sigma_min, decimals=3)
            
            
            if sigma_max <= 0:
                qualified = True
            
            else:
                rho = sigma_min / sigma_max
                if rho >= -1:
                    gamma_n = self.get_gamma_n(unit)
                    if gamma_d * gamma_n * (sigma_max - sigma_min) < gamma_t * sigma_0:
                        qualified = True
                        fatigue_surplus = (gamma_d * gamma_n * (sigma_max - sigma_min) / gamma_t - sigma_0) / sigma_0
                    else:
                        qualified = False

                else: # rho < -1
                    gamma_rho = self.get_gamma_rho(rho)
                    if gamma_d * gamma_n_prime * sigma_max <= gamma_t * gamma_rho * sigma_0:
                        qualified = True
                        fatigue_surplus = (gamma_d * gamma_n_prime * sigma_max / (gamma_t * gamma_rho) - sigma_0) / sigma_0
                    else:
                        qualified = False

            
            unit.fatigue_qualified = qualified
            unit.fatigue_surplus = fatigue_surplus
        
        print('单元%d 疲劳检算完成.' % (unit.num))
        
        
        
        
    def show_fatigue_check_results(self):
        print('### 疲劳检算 ###\n单元\t合格\t富余率')
        for unit in self.units.values():
            qualified_info = '%2d\t%s\t' % (unit.num, '是' if unit.fatigue_qualified else '否')
            if unit.surplus is not None:
                qualified_info += '%.3f' % (unit.surplus)
            else:
                qualified_info += '\\'
            print(qualified_info)

            
############# 强度检算 ##############            
     
    def strength_check(self, unit, params):
        '''
        
        '''

        sigma = 200 * 10**3  # KPa
        qualified = None
        strength_surplus = None
        
        N = self.N 
        A_j = self.A_j

        qualified = True if (abs(N) / A_j < sigma).all() else False
        strength_surplus = (np.max(np.abs(N)) / A_j - sigma) / sigma
        
        unit.strength_qualified = qualified
        unit.strength_surplus = strength_surplus
        
        print('单元%d 强度检算完成.' % (unit.num))        
        

    def show_strength_check_results(self):
        print('### 强度检算 ###\n单元\t合格\t富余率')
        for unit in self.units.values():
            qualified_info = '%2d\t%s\t%.3f' % (unit.num, '是' if unit.strength_qualified else '否', unit.strength_surplus)
            print(qualified_info)

            
############### 刚度检算 ##################
    def stiffness_check(self):
        ''''''
        qualities = []
        for unit in self.bridge.units.values():
            qualified = None
            
            lambda_x, lambda_y = unit.lambda_
            
            N = unit.N
            if min(N) >= 0 and unit.num not in (self.top_chord_units_nums + self.bottom_chord_nodes_nums):
                thresh = (0, 180)
            else:
                thresh = (0, 100)

            # if lambda_x < thresh and lambda_y < thresh:
            if thresh[0] < lambda_x < thresh[1] and thresh[0] < lambda_y < thresh[1]:
                qualified = True
            else:
                qualified = False
            
            #unit.stiffness_qualified = qualified
            qualities.append(qualified)

            #print('单元%d 刚度检算完成.' % (unit.num))
            # if unit.num == 10:
            #     print(lambda_x, lambda_y)

        return np.array(qualities).all()


    def show_stiffness_check_results(self):
        print('### 刚度检算 ###\n单元\t合格')
        for unit in self.units.values():
            qualified_info = '%2d\t%s' % (unit.num, '是' if unit.stiffness_qualified else '否')
            print(qualified_info)


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

        phi_1x = interp1d(np.array(lambda_X), np.array(lambda_X), 'linear')(lambda_x)

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

        phi_1y = interp1d(np.array(lambda_Y), np.array(lambda_Y), 'linear')(lambda_y)

        return phi_1y


    
    def overall_stability_check(self, unit, params):
        '''整体稳定检算'''


        sigma = 200 * 10**3  # KPa
        
        
        qualified = None
        surplus = None


        
        N = self.N
        A_m = self.A_m

        lambda_x, lambda_y = self.lambda_
        phi_1 = min(get_phi_1x(lambda_x), get_phi_1y(lambda_y))

        if min(N) > 0:
            qualified = True
        else:
            qualified = (abs(min(N)) / A_m) <= (phi_1 * sigma)

        surplus = (N / (A_m * phi_1) - sigma) / sigma

        unit.overall_stability_qualified = qualified
        unit.overall_stability_surplus = surplus

        print('单元%d 整体稳定检算完成.' % (unit.num))


    def show_overall_stability_check_results(self):
        print('### 整体稳定检算 ###\n单元\t合格')
        for unit in self.units.values():
            qualified = unit.overall_stability_qualified
            qualified_info = '%2d\t%s' % (unit.num, '是' if qualified else '否')
            print(qualified_info)
            

############### 局部稳定检算 ##################
    def local_stability_check(self):

        qualities = []
        for unit in self.bridge.units.values():
            qualified = None
            surplus = None
            
            if unit.is_zero_bar:
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
            qualities.append(qualified)
                

        #unit.local_stability_qualified = qualified

        #print('单元%d 局部稳定检算完成.' % (unit.num))
        return np.array(qualities).all()

    
    def show_local_stability_check_results(self):
        print('### 局部稳定检算 ###\n单元\t合格')
        for unit in self.units.values():
            qualified = unit.local_stability_qualified
            qualified_info = '%2d\t%s' % (unit.num, '是' if qualified else '否')
            print(qualified_info)



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



################ 检算入口 #######################

    def check(self, checking=True):
        self._checking = checking
        for unit in self.units.values():
            unit._checking = checking

       

    def check_unit(self, unit, which=None, params=None):

        assert which is not None, '请指定检算类型.'
        assert self._checking, '当前不在检算状态.'
        
        if which == 0:
            
            self.fatigue_check(unit)

        elif which == 1:
            
            self.strength_check(unit)

        elif which == 2:
            
            self.stiffness_check(unit)

        elif which == 3:
            
            self.overall_stability_check(unit)

        elif which == 4:
            
            self.local_stability_check(unit)


    def set_section_params(self, units_nums_group, beam_section_data):
        
        assert self._checking, '当前不在检算状态.'

        for unit_num in units_nums_group:
            unit = self.units[unit_num]
            unit.beam_section_data = beam_section_data

        #print('当前修改参数杆件组\n', units_nums_group)



        
    def update(self):
        '''更新与横截面参数有依赖关系的量'''

        for unit in self.units.values():
            unit.area
            unit.kij

        self.K
        self.reduced_K
        self._checking = False
        for unit in self.units.values():
            unit._checking = False

        self.get_nodes_vdisps()
        self.get_units_axial_forces()
        self.get_worst_cases_load()
        self.get_worst_cases_disps()





    def check_group(self, unit_nums_group, which=None, params=None):
        '''
        params - 某种检算人工输入的参数
        '''
        
        assert which is not None, '请指定检算类型.'
        assert self._checking, '当前不在检算状态.'
        

        for unit_num in unit_nums_group:
            unit = self.units[unit_num]
            self.check_unit(unit, which, params)

            unit._checking = False
            

    
    def get_something_for_checking(self, which):
        which_types = { 0: '疲劳', 1: '强度', 2: '刚度', 
                        3: '整体稳定', 4: '局部稳定', 5: '梁体毛度'}
        types2attr =  { '疲劳': 'fatigue', '强度': 'strength', '刚度': 'stiffness', 
                         '整体稳定': 'overall_stability', '局部稳定': 'local_stability', '梁体毛度':''}


        return types2attr[which_types[which]]


    def show_check_group_results(self, unit_nums_group, which=None):


        title = '### %s 检算 ###\n单元\t合格\t富余率' % (which_types[which])
        print(title)
        for unit_num in unit_nums_group:
            unit = self.units[unit_num]
            qualified = getattr(unit, get_something_for_checking(which) + '_qualified')
            surplus = getattr(unit, types2attr[which_types[which]] + '_surplus')
            qualified_info = '%2d\t%s\t%.3f' % (unit.num, 
                                                '是' if qualified else '否', 
                                                surplus if surplus else '/')
            
            print(qualified_info)
    

    # def check_all(self, which=None):
    #     assert which is not None, '请指定检算类型.'
    #     assert self._checking, '当前不在检算状态.'

    #     which_types = { 0: '疲劳', 1: '强度', 2: '刚度', 
    #                     3: '整体稳定', 4: '局部稳定', 5: '梁体毛度'}        

    #     for unit_num in unit_nums_group:
    #         unit = self.units[unit_num]
    #         self.check_unit(unit, which)

    #         unit._checking = False

    # def show_check_results(self, which=None):

    #     assert which is not None, '请指定检算类型.'
        
    #     which_types = { 0: '疲劳', 1: '强度', 2: '刚度', 
    #                     3: '整体稳定', 4: '局部稳定', 5: '梁体毛度'}

    #     types2attr =  { '疲劳': self.fatigue_check, '强度': self.strength_check, '刚度': 'stiffness', 
    #                     '整体稳定': 'overall_stability', '局部稳定': 'local_stability', '梁体毛度':''}

    #     title = '### %s 检算 ###\n单元\t合格\t富余率' % (which_types[which])
    #     print(title)
    #     for unit in self.units.values():
    #         qualified = getattr(unit, types2attr[which_types[which]] + '_qualified')
    #         surplus = getattr(unit, types2attr[which_types[which]] + '_surplus')
    #         qualified_info = '%2d\t%s\t%.3f' % (unit.num, 
    #                                             '是' if qualified else '否', 
    #                                             surplus if surplus else '/')
            
    #         print(qualified_info)


##########################################



