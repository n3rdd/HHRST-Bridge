from collections import OrderedDict
from math import sqrt, sin, cos, tan, atan, pi, floor
import os

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy import interpolate

from utils import partition_kij, update_K, reshape_K, reduce_K, pretty_print




######################################
#                                    #
#               节点类                #
#                                    #
######################################

class Node:
    '''杆件单元节点类'''
    
    def __init__(self, num):
        self.num = num
        self.coordinates = (None, None)
        self.vdisps = []
        
    def __repr__(self):
        return 'Node(%d)' % (self.num)


    

    
######################################
#                                    #
#               杆件单元类             #
#                                    #
######################################    
class Unit:
    '''杆件单元类'''
    def __init__(self, num, nodes_pair):
        self.num             =    num
        self.nodes_pair      =    nodes_pair
        self.node_i          =    self.nodes_pair[0]
        self.node_j          =    self.nodes_pair[1]
        
        self._length  =   None
        self._alpha   =   None
        self._E       =   None

        
        # 杆件类型
        self._type = None

        # 梁截面数据([腹板宽度 翼缘厚度 腹板厚度 翼缘宽度])
        self.beam_section_data = []
        self._area    =   None
        self._kij     =   None
        self.axial_forces = []
        

        # 检算
        self.fatigue_qualified = None
        self.strength_qualified = None
        self.stiffness_qualified = None
        self.overall_stability_qualified = None
        self.local_stability_qualified = None
        self.deflection_qualified = None         # 挠度


        # 富余度
        self.fatigue_surplus = None
        self.strength_surplus = None
        self.stiffness_surplus = None
        self.overall_stability_surplus = None
        self.local_stability_surplus = None
        
        

        self._checking = False


    def set_section_params(self, beam_section_data):
        self.beam_section_data = beam_section_data
        

    def check(self, checking=True):
        self._checking = checking

        
    @property
    def alpha(self):
        if self._alpha:
            return self._alpha
        
        i, j = self.node_i.num, self.node_j.num
        xi, yi = self.node_i.coordinates
        xj, yj = self.node_j.coordinates
        if xi == xj:
            alpha = pi / 2
        else:
            alpha = atan((yj - yi) / (xj - xi)) 
        
        self._alpha = alpha
        return self._alpha
    
    
    @property
    def length(self):
        if self._length:
            return self._length
        
        i, j = self.node_i.num, self.node_j.num
        xi, yi = self.node_i.coordinates
        xj, yj = self.node_j.coordinates
        
        self._length = sqrt((yj - yi)**2 + (xj - xi)**2)
        return self._length
        
    
    
    def get_area(self):
        b2, t1, t2, b1 = self.beam_section_data
        area = b2 * t2 + 4 * b1 * t1
        
        return area

    @property
    def area(self):
        '''截面面积'''
        # 梁截面数据([腹板宽度 翼缘厚度 腹板厚度 翼缘宽度])
        #             b2       t1      t2     b1

        if not self._area:
            self._area = self.get_area()

        elif self._checking:
            self._area = self.get_area()
            print('检算 => 单元 %d 截面面积已计算.' % (self.num))

        return self._area

    
    def get_kij(self):
        a = self.alpha
        l = self.length       # 单元长度

        M = np.array(
        [[cos(a)**2, cos(a)*sin(a), -cos(a)**2, -cos(a)*sin(a)],
        [cos(a)*sin(a), sin(a)**2, -cos(a)*sin(a), -sin(a)**2],
        [-cos(a)**2, -cos(a)*sin(a), cos(a)**2, cos(a)*sin(a)],
        [-cos(a)*sin(a), -sin(a)**2, cos(a)*sin(a), sin(a)**2]])
        
        A = self.area
        E = self._E
        
        kij = E * A / l * M

        return kij


    @property
    def kij(self):
        '''单元刚度矩阵'''
        if self._kij is None:
            self._kij = self.get_kij()

        elif self._checking:
            self._kij = self.get_kij()
            print('检算 => 单元 %d 单元刚度矩阵已重新计算.' % (self.num))

        return self._kij
        
    
    @property
    def A_m(self):
        '''毛截面积'''
        b2, t1, t2, b1 = self.beam_section_data
        A_m = (4 * b1 * t1 + (b2 + 2 * t1) * t2) * 10**(-6) # m**2
        return A_m


    @property
    def delta_A(self):
        '''栓孔面积'''
        _, t1, _, _ = self.beam_section_data
        delta_A = 8 * 23 * t1 * 10**(-6)
        return delta_A


    @property
    def A_j(self):
        '''净面积'''
        return self.A_m - self.delta_A


    @property
    def I(self):
        '''极惯性矩'''
        b2, t1, t2, b1 = self.beam_section_data
        # m**4
        I_x = (1 / 12 * (2 * b1 + t2) * (2 * t1 + b2) ** 3 - 2 * (1 / 12) * b1 * b2 ** 3) * 10 ** (-12)
        I_y = (1 / 12 * b2 * t2 ** 3 + 2 * (1 / 12) * t1 * (2 * b1 +  t2) ** 3) * 10 ** (-12)

        return I_x, I_y


    @property
    def r(self):
        '''回转半径'''
        I_x, I_y = self.I
        A_m = self.A_m

        r_x = sqrt(I_x / A_m)
        r_y = sqrt(I_y / A_m)

        return r_x, r_y
    
    @property
    def lambda_(self):
        '''长细比'''

        l = self.length
        if self._type in [0, 1]: # 上下弦杆
            l_0x = l
            l_0y = l

        elif self._type == 2:  # 端斜杆, 端立杆, 中间支点处
            l_0x = l
            l_0y = 0.9 * l

        else:
            l_0x = l
            l_0y = 0.8 * l

        r_x, r_y = self.r
        lambda_x = l_0x / r_x
        lambda_y = l_0y / r_y

        return lambda_x, lambda_y



    @property
    def N_k(self):
        '''最大活载''' 
        
        return self.max_force  # [pos_max, neg_min]


    @property
    def N(self):
        N_k = np.array(self.N_k)
        mu = 1 + 28 / (40 + self.length) - 1
        N = (1 + mu) * N_k + self.h
        return N
    


    def __repr__(self):
        return 'Unit(%d, (%d, %d))' % (self.num, self.nodes_pair[0].num, self.nodes_pair[1].num)
    


######################################
#                                    #
#               活载类                #
#                                    #
######################################

class Force:
    def __init__(self, size):
        self.size = size


class Load(Force):
    def __init__(self, type_='zk'):
        self.type_ = type_

class ZkLoad(Load):
    def __init__(self):
        self.conc_load = None
        self.uniform_load = None
        self.interval = None

######################################
#                                    #
#               桥类                  #
#                                    #
######################################

class Bridge:
    
    def __init__(self):
        self.nodes_nums =    []
        self.nodes      =    OrderedDict()   # {node_num: node}

        self.units      =    OrderedDict()   # {unit_num: unit}
        self.units_nums =    []


        self._length    =    None            # 长度/米
        self._K         =    None
        self._reduced_K =    None
        self._n_nodes   =    None
        self._E         =    None

        self.load       =    None

        self._checking   =   False           # 检算状态


        self._bottom_chord_nodes_nums = []    # 下弦杆节点编号

        # 杆件类型
        # 0 - 上弦杆
        # 1 - 下弦杆
        # 2 - 端斜梁、端立杆、中间支点处立柱
        self.top_chord_units_nums = []
        self.bottom_chord_units_nums = []
        self.side_units_nums = []
        self.other_units_nums = []


        self.deflection_qualified = None       # 梁体挠度




        
####################### 读入数据 #####################
    def load_nodes_data(self, path):
        ''' 读入节点坐标数据 '''
        
        file_name = 'nodes_coordinates_' + str(self.length) + '.txt'
        file_path = os.path.join(path, 'nodes', file_name)
        nodes_nums = []
        with open(file_path, 'r') as nodes_coordinates_file:
            for line in nodes_coordinates_file.readlines():
                node_num, x, y = [int(string) for string in line.strip().split()]
                nodes_nums.append(node_num)
                node = Node(node_num)
                node.coordinates = (x, y)
                self.nodes[node_num] = node

            self.nodes_nums = nodes_nums
                
    
    def load_units_data(self, path):
        '''读入杆件数据'''
        
        # 读入单元编号对应杆件编号对
        file_name = 'unit_node_mapping_' + str(self.length) + '.txt'
        file_path = os.path.join(path, 'units', file_name)
        with open(file_path, 'r') as unit_nodes_mapping_file:
            units_nums = []
            for line in unit_nodes_mapping_file.readlines():
                unit_num, i, j = [int(string) for string in line.strip().split()]
                units_nums.append(unit_num)
                node_i, node_j = self.nodes[i], self.nodes[j]
                nodes_pair = (node_i, node_j)
                unit = Unit(unit_num, nodes_pair)
                self.units[unit_num] = unit
            
            self.units_nums = units_nums
        
        # 读入梁截面数据
        file_name = 'beam_section_data_' + str(self.length) + '.txt'
        file_path = os.path.join(path, 'units', file_name)
        with open(file_path, 'r') as beam_section_data_file:
            for line in beam_section_data_file.readlines():
                # 单元编号 腹板宽度 翼缘厚度 腹板厚度 翼缘宽度
                beam_section_data = line.strip().split()
                self.units[int(beam_section_data[0])].beam_section_data = [float(string) for string in beam_section_data[1:]]

        
        # 读入杆件类型
        # 0 - 上弦杆
        # 1 - 下弦杆
        # 2 - 端斜梁、端立杆、中间支点处立柱
        # 3 - 其他
        file_name = 'units_types_data_' + str(self.length) + '.txt'
        file_path = os.path.join(path, 'units', file_name)
        with open(file_path, 'r') as units_types_data_file:
            not_other_units_nums = []
            for line in units_types_data_file.readlines():
                if line[0] == '#' or line[0].isspace():
                    continue
                curr_units_nums = list(map(int, line.strip().split()))
                type_num = curr_units_nums.pop(0)

                not_other_units_nums += curr_units_nums
                
                if type_num == 0:
                    self.top_chord_units_nums = curr_units_nums
                elif type_num == 1:
                    self.bottom_chord_units_nums = curr_units_nums
                elif type_num == 2:
                    self.side_units_nums = curr_units_nums

                for unit_num in curr_units_nums:
                    self.units[int(unit_num)].type_ = type_num

            other_units_nums = set(self.units_nums) - set(not_other_units_nums)
            for unit_num in other_units_nums:
                self.units[unit_num].type_ = 3
            self.other_units_nums = other_units_nums
                

                
    def load_data(self, path):
        
        self.__init__()
        
        self.load_nodes_data(path)
        self.load_units_data(path)
        
        
    def load_params(self, **kargs):
        '''载入参数
            E - 杨氏模量 (Pa)
            P - 外力    (kN)
            h - 恒载    (kN)
            
        '''
        self._E = kargs['E']
        for unit_num, unit in self.units.items():
            unit._E = self._E
        
        self.P = kargs['P']
        self.h = kargs['h']
        
       
        self.bottom_chord_nodes_nums = kargs['bottom_chord_nodes_nums']
        self.bottom_chord_nodes_indices = list(range(len(self.bottom_chord_nodes_nums)))
        self.bottom_chord_length = kargs['bottom_chord_length']
        self.load = kargs['load']

                  



#################### 属性 ########################

    @property
    def length(self):
        '''桥长度'''
        return self._length
        
       
    
    @property
    def n_nodes(self):
        '''结点个数'''
        if self._n_nodes:
            return self._n_nodes
        
        self._n_nodes = len(self.nodes.keys())
        return self._n_nodes
        
    

    def get_K(self):
        K = np.zeros((self.n_nodes, self.n_nodes, 2, 2))
        for unit_num, unit in self.units.items():
            K = update_K(K, unit, unit.kij)
            
        return reshape_K(K, self.n_nodes)

    @property
    def K(self):
        '''总体刚度矩阵'''
        if self._K is None:
            self._K = self.get_K()

        elif self._checking:
            self._K = self.get_K()
            print('检算 => 总体刚度矩阵已重新计算.')

        
        return self._K
    
    @property
    def reduced_K(self):
        if self._reduced_K is None:
            self._reduced_K = reduce_K(self.K, self.length)
        
        elif self._checking:
            self._reduced_K = reduce_K(self.K, self.length)
            print('检算 => 缩减总体刚度矩阵已重新计算.')
            self._checking = False

        return self._reduced_K
    

    


######################## 计算节点竖向位移 ####################

    def save_nodes_vdisps_moment(self, nodes_vdisps_moment):
        '''将[某一]时刻[所有]节点的竖向位移保存到各个节点的竖向位移向量中
        随桥结构变化'''
        
        # 子类中实现
        raise NotImplementedError
    
    
    def get_nodes_vdisps_moment_on_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点上][所有]节点的竖向位移
        随桥结构变化'''
        
        # 子类中实现
        raise NotImplementedError
    
    
    def get_nodes_vdisps_moment_between_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点间][所有]节点的竖向位移
        随桥结构变化'''
        
        # 子类中实现
        raise NotImplementedError
    
    
    
    def get_nodes_vdisps_moment(self, point):
        '''计算力移动过程中，[某一]时刻[所有]节点的竖向位移'''
        '''
        bc_indices   0                         1                           2   ...    7                             8
        bc_nodes     1                         3                           5   ...   15                            16
        bc_axis     0.0 0.1 0.2 ...  7.8 7.9  8.0 8.1 8.2 ...  15.8 15.9 16.0  ...  56.0 56.1 56.2 ... 63.8 63.9  64.0
        '''
        
        #bc_nodes_nums = self.bottom_chord_nodes_nums
        bc_nodes_indices = self.bottom_chord_nodes_indices
        bc_length = self.bottom_chord_length
        
        # 力作用在节点上
        bc_nodes_corr_force_disps = [bc_length * bc_node_index for bc_node_index in bc_nodes_indices]  # 力作用在下弦杆某一节点时对应的位移
        if int(point) == point and point in bc_nodes_corr_force_disps:
            D = self.get_nodes_vdisps_moment_on_nodes(point)

        # 力作用在节点间
        else:
            D = self.get_nodes_vdisps_moment_between_nodes(point)

        nodes_vdisps_moment = D
        return nodes_vdisps_moment
    
    
    def nodes_vdisps_init(self):
        '''将节点竖向位移清空'''
        for node in self.nodes.values():
            node.vdisps.clear()
    
    def get_bc_axis(self):
        '''为下弦杆建立坐标轴，力在其上移动'''
        
        start, end = 0, (len(self.bottom_chord_nodes_indices) - 1) * self.bottom_chord_length
        bc_axis = np.arange(start, end + 0.1, step=0.1)
        
        return bc_axis
    
    
    def get_nodes_vdisps(self):
        '''计算力移动过程中，所有时刻所有节点的竖向位移'''
        '''
        bc_indices  0   1   2   3   4   5   6   7   8
        bc_nodes    1   3   5   7   9  11  13  15  16
        bc_axis     0   8  16  24  32  40  48  56  64
        '''
        
        self.nodes_vdisps_init()
        
        bc_axis = self.get_bc_axis()
        
        # 力在下弦杆移动
        for point in bc_axis:
            nodes_vdisps_moment = self.get_nodes_vdisps_moment(point)
            self.save_nodes_vdisps_moment(nodes_vdisps_moment)
     
        print('所有节点竖向位移已计算完毕.')
    

    
    def show_nodes_vdisps(self):
        '''节点竖向位移影响线'''
        for node in self.nodes.values():
            x = np.arange(0, self.length + 0.1, 0.1)
            y = np.array(node.vdisps)
            plt.plot(x, y)
            plt.xlabel(u'单位力位移')
            plt.ylabel(u'节点竖向位移')
            plt.title(u'节点 ' + str(node.num) + u' 影响线')
            #plt.title('V' + str(node))
            #plt.savefig('/Users/nerd/Desktop/figs/%s.png' % ('V' + str(node)))
            plt.show()

            
            

############## 计算杆件单元轴力 #################

    def units_axial_forces_init(self):
        for unit in self.units.values():
            unit.axial_forces.clear()

            
    def save_units_axial_forces_moment(self, units_axial_forces_moment):
        for unit, one_unit_axial_force_moment in zip(self.units.values(), units_axial_forces_moment):
            unit.axial_forces.append(one_unit_axial_force_moment)

            
    def get_one_unit_axial_force_moment(self, unit, nodes_vdisps_moment):
        '''计算某一时刻某一杆件单元的轴力
        随桥结构变化'''
        
        raise NotImplementedError
    
    
    def get_units_axial_forces_moment(self, point):
        '''计算力移动过程中，某一时刻所有杆件单元的轴力'''
        
        nodes_vdisps_moment = self.get_nodes_vdisps_moment(point)
        units_axial_forces_moment = []
        for unit in self.units.values():
            one_unit_axial_force_moment = self.get_one_unit_axial_force_moment(unit, nodes_vdisps_moment)
            units_axial_forces_moment.append(one_unit_axial_force_moment)   # 需检查和self.units顺序是否对应
            
        return units_axial_forces_moment
    
    
    def get_units_axial_forces(self):
        '''计算力移动过程中，所有时刻所有杆件单元的轴力'''
        '''
        bc_indices  0   1   2   3   4   5   6   7   8
        bc_nodes    1   3   5   7   9  11  13  15  16
        bc_axis     0   8  16  24  32  40  48  56  64
        '''
        
        self.units_axial_forces_init()
        
        bc_axis = self.get_bc_axis()
        
        # 力在下弦杆移动
        for point in bc_axis:
            units_axial_forces_moment = self.get_units_axial_forces_moment(point)
            self.save_units_axial_forces_moment(units_axial_forces_moment)
    
        print('所有杆件单元轴力已计算完毕.')

        # 判断零杆
        is_zero_bar = None
        for unit in self.units.values():
            unit_axial_forces = np.array(unit.axial_forces)
            if (np.around(unit_axial_forces, decimals=8) == 0).all():
                is_zero_bar = True
            else:
                is_zero_bar = False

            unit.is_zero_bar = is_zero_bar

        print('所有零杆已标识.')

    
    def show_zero_bars(self):
        for unit in self.units.values():
            if unit.is_zero_bar:
                print(unit.num, end=' ')
        print()
    
    
    def show_units_axial_forces(self):
        '''杆件单元轴力影响线'''
        for unit in self.units.values():
            x = np.arange(0, self.length + 0.1, 0.1)
            y = np.array(np.around(unit.axial_forces, 3))
            plt.plot(x, y)
            plt.xlabel(u'单位力位移')
            plt.ylabel(u'杆件轴力')
            plt.title(u'杆件 ' + str(unit.num) + u' 影响线')
            #plt.title('V' + str(node))
            #plt.savefig('/Users/nerd/Desktop/figs/%s.png' % ('V' + str(node)))
            plt.show()

    

    


##################### 计算最不利位移/荷载 #####################
    
    
    def get_worst_cases(self, data, load):
        '''计算每根杆件最不利荷载/每个节点最不利位移'''

        pos_data = (data > 0) * data
        neg_data = (data < 0) * data

        pos_max, neg_min = -np.inf, np.inf

        i = -1
        while True:
            selected = load[int(i - (self.length / 0.1 + 1)): i]

            pos_sum = (pos_data * selected).sum()
            neg_sum = (neg_data * selected).sum()

            pos_max = max(pos_max, pos_sum)
            neg_min = min(neg_min, neg_sum)

            if int(i - (self.length / 0.1 + 1)) == -load.shape[0]:
                break

            i -= 1

        return pos_max, neg_min



    def get_worst_cases_load(self):
        '''计算所有杆件单元最不利荷载'''
        for unit in self.units.values():
            unit_axial_forces = np.array(unit.axial_forces)
            load = np.array(self.load)
            #pos_max, neg_min = self.get_unit_worst_cases_load(unit_axial_forces, load, pos_max=-np.inf, neg_min=np.inf)
            pos_max, neg_min = self.get_worst_cases(unit_axial_forces, load)     
            unit.max_force = (pos_max, neg_min)
            
        print('所有杆件单元最不利荷载已计算完毕.')



    def get_worst_cases_disps(self):
        '''计算所有杆件单元最不利位移'''
        for node in self.nodes.values():
            node_vdisps = np.array(node.vdisps)
            load = np.array(self.load)
            #pos_max, neg_min = self.get_unit_worst_cases_load(unit_axial_forces, load, pos_max=-np.inf, neg_min=np.inf)
            pos_max, neg_min = self.get_worst_cases(node_vdisps, load)     
            node.max_vdisps = (pos_max, neg_min)
            
        print('所有杆件单元最不利位移已计算完毕.')


################### 检算 ##############################
### 警告：以下内容较繁琐且可读性较差，具体细节请参照规范！  ###
######################################################




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
        gamma_d       = 1              
        gamma_t       = 1
        gamma_n       = None
        gamma_n_prime = 1
        sigma_0       = 110300
        
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
    def stiffness_check(self, unit, params):
        ''''''


        stiffness_qualified = None
        
        # 需要判断斜杆
        l_0x = unit.length  # 计算长度
        l_0y = unit.length  
        r_x, r_y= self.r    # 回转半径

        # 人工输入lambda_x, lambda_y
        lambda_x = l_0x / r_x
        lambda_y = l_0y / r_y
        
        N = self.N
        if min(N) >= 0 and unit.num not in (self.top_chord_units_nums + self.bottom_chord_nodes_nums):
            thresh = 180
        else:
            thresh = 100

        if lambda_x < thresh and lambda_y < thresh:
            stiffness_qualified = True
        else:
            stiffness_qualified = False
        
        unit.stiffness_qualified = stiffness_qualified

        print('单元%d 刚度检算完成.' % (unit.num))


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
        ''''''


        sigma = 200 * 10**3  # KPa
        
        
        overall_stability_qualified = None
        overall_stability_surplus = None


        if unit.is_zero_bar:
            overall_stability_qualified = True

        else:
            N = self.N
            A_m = self.A_m

            if min(N) > 0:
                overall_stability_qualified = True
            else:
                lambda_x, lambda_y = self.lambda_
                phi_1 = min(get_phi_1x(lambda_x), get_phi_1y(lambda_y))
                overall_stability_qualified = (abs(min(N)) / A_m) <= (phi_1 * sigma)

        unit.overall_stability_qualified = overall_stability_qualified

        print('单元%d 整体稳定检算完成.' % (unit.num))


    def show_overall_stability_check_results(self):
        print('### 整体稳定检算 ###\n单元\t合格')
        for unit in self.units.values():
            qualified = unit.overall_stability_qualified
            qualified_info = '%2d\t%s' % (unit.num, '是' if qualified else '否')
            print(qualified_info)
            

############### 局部稳定检算 ##################
    def local_stability_check(self, unit, params):
        qualified = None
        surplus = None
        
        if unit.is_zero_bar:
            qualified = True
        else:
            N = self.N
            if min(N) > 0:
                qualified = True
            
            else:
                b2, t1, t2, b1 = unit.beam_section_data
                lambda_x, lambda_y = unit.lambda_


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

        print('单元%d 局部稳定检算完成.' % (unit.num))

    
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

    def __repr__(self):
        units_repr = [str(unit) for unit in self.units.values()]
        return '[' + '\n'.join(units_repr) + ']'
    
    


    
    


    

    
### 计算下弦杆结点位移 ###
'''
下弦杆共9个节点(1, 3, 5, 7, ... 15, 16)，
对于其中2个节点(1, 16):
因每一时刻节点1和16竖向位移都为0， 所以V1, V16为零向量

对于另外7个节点(3, 5, 7, ..., 15):
第1和第9时刻力作用在节点1和16，每个节点竖向位移为0，
其余时刻分别令7个F中(y3, y5, y7, ... y15)为-P，其余元素为0，
由矩阵方程得到对应9个时刻的9个29维位移向量D，
分别取出9个时刻位移向量D的竖向位移v2, v3, ..., v15得到
7个节点对应9个时刻的竖向位移向量V2, V3, ... V15
'''
'''

                  0   1   2   3   4   5   6   7  ...       27   28
     F  x1  y1  [x2  y2  x3  y3  x4  y4  x5  y5  ... x15  y15  x16] y16

        u1  v1   u2  v2  u3  v3  u4  v4  u5  v5  ... u15  v15  u16  v16 
1    D   0   0  [ 0   0   0   0   0   0   0   0   ...  0    0    0]   0
3    D   0   0  [ 0   #   0   #   0   #   0   #        0    #    0]   0     <=  y3 = -P
5    D   0   0  [ 0   #   0   #   0   #   0   #        0    #    0]   0     <=  y5 = -P
   ...
15   D   0   0  [ 0   #   0   #   0   #   0   #        0    #    0]   0     <=  y15 = -P
16   D   0   0  [ 0   0   0   0   0   0   0   0   ...  0    0    0]   0
            V1       V2      V3      V4      V5           V15       V16   

          y3 = -P  y5 = -P  ... y15 = -P 
V1  [0       0        0     ...     0      0]   9 * 1
V2  [0      v2       v2     ...    v2      0]
...
V15 [0     v15      v15     ...   v15      0]
V16 [0       0        0     ...     0      0]
'''