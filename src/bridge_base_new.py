from collections import OrderedDict
from math import sqrt, sin, cos, tan, atan, pi, floor
import json
import os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

import numpy as np
np.set_printoptions(precision=3, suppress=True)
# from scipy.interpolate import interp1d

from utils import partition_kij, seat_kij, reshape_K, reduce_K, pretty_print




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
        self.h        =   None

        
        # 杆件类型
        self._type = None

        # 梁截面数据([腹板宽度 翼缘厚度 腹板厚度 翼缘宽度])
        self.section_params = []
        self._A_m    =   None   # 毛面积
        self._kij     =   None  # 单元刚度矩阵

        self.axial_forces = []  # 轴力

        self.max_forces = (None, None)
        

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


    def set_section_params(self, section_params):
        self.section_params = section_params
        

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
        
    
    
    def get_A_m(self):
        b2, t1, t2, b1 = self.section_params
        A_m = 4 * b1 * t1 + (b2 + 2 * t1) * t2
        
        return A_m

    @property
    def A_m(self):
        '''毛面积'''
        # 梁截面数据([腹板宽度 翼缘厚度 腹板厚度 翼缘宽度])
        #             b2       t1      t2     b1

        if self._A_m is None:
            self._A_m = self.get_A_m()

        return self._A_m

    
    def get_kij(self):
        a = self.alpha
        length = self.length       # 单元长度

        M = np.array(
        [[cos(a)**2, cos(a)*sin(a), -cos(a)**2, -cos(a)*sin(a)],
        [cos(a)*sin(a), sin(a)**2, -cos(a)*sin(a), -sin(a)**2],
        [-cos(a)**2, -cos(a)*sin(a), cos(a)**2, cos(a)*sin(a)],
        [-cos(a)*sin(a), -sin(a)**2, cos(a)*sin(a), sin(a)**2]])
        
        A = self.A_m
        E = self._E
        
        kij = E * A / length * M

        return kij


    @property
    def kij(self):
        '''单元刚度矩阵'''
        if self._kij is None:
            self._kij = self.get_kij()

        return self._kij
        
    


    
    ######## 检算 ########
    @property
    def delta_A(self):
        '''栓孔面积'''
        _, t1, _, _ = self.section_params
        delta_A = 8 * 23 * t1 * 10 ** (-3)
        return delta_A


    @property
    def A_j(self):
        '''净面积'''
        return self.A_m - self.delta_A


    @property
    def I(self):
        '''极惯性矩'''
        b2, t1, t2, b1 = self.section_params
        # m**4
        I_x = (1 / 12 * (2 * b1 + t2) * (2 * t1 + b2) ** 3 - 2 * (1 / 12) * b1 * b2 ** 3)
        I_y = (1 / 12 * b2 * t2 ** 3 + 2 * (1 / 12) * t1 * (2 * b1 +  t2) ** 3)

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
        axial_forces = np.around(self.axial_forces, decimals=3)
        pos_max, neg_min = self.max_forces
        pos_max += (((axial_forces < 0) * axial_forces) * 1.5).sum()
        neg_min += (((axial_forces > 0) * axial_forces) * 1.5).sum()
        
        return pos_max, neg_min


    @property
    def N_kf(self):
        '''疲劳最大活载'''
        axial_forces = np.around(self.axial_forces, decimals=3)
        pos_max, neg_min = self.N_k
        pos_max += (((axial_forces < 0) * axial_forces) * 6.4).sum()
        neg_min += (((axial_forces > 0) * axial_forces) * 6.4).sum()
        
        return pos_max, neg_min
    

    @property
    def N(self):
        '''最大内力'''
        N_k = np.array(self.N_k)
        mu = 1 + 28 / (40 + self.length) - 1
        N = (1 + mu) * N_k + self.h

        return N
    


    def __repr__(self):
        return 'Unit(%d, (%d, %d))' % (self.num, self.nodes_pair[0].num, self.nodes_pair[1].num)
    

######################################
#                                    #
#               支座类                #
#                                    #
######################################
class Support:
    def __init__(self, node_num, h=False, v=False):
        self.node_num = node_num
        self.h = h  # 横向约束
        self.v = v  # 竖向约束


######################################
#                                    #
#               活载类                #
#                                    #
######################################

class Load:
    def __init__(self):
        self.load = None

    def __call__(self):
        return self.load

class ZkLoad(Load):
    '''zk活载'''
    '''
    步长 = 0.1m

           6.4      0             200    
         均布荷载   填充          集中荷载                     均布荷载
    {| | | | | |} . . . {#  .  #  .   #  .  #}  . . .  {| | | | | |}
        桥长度       0.8    1.6     1.6   1.6       0.8   桥长度      
    '''
    def __init__(self, uniform_len):
        uniform_load = [6.4 for i in range(int(uniform_len / 0.1) + 1)] # 均布荷载，每个力间隔0.1m
        padding = [0 for i in range(int(0.8 / 0.1) - 1)] # 填充0

        # 集中荷载，每个间隔0.8m，与均布荷载相距0.8m
        conc_load = [200] + padding + [0] + padding + [200] + padding + \
                    [0] + padding + [200] + padding + [0] + padding + [200]

        load = uniform_load + padding + conc_load + padding + uniform_load

        outside_padding = [0 for i in range(int(uniform_len / 0.1) + 1)]
        load = outside_padding + load + outside_padding

        # (size,) array
        self.load = np.array(load)

######################################
#                                    #
#               桥类                  #
#                                    #
######################################

class Bridge:
    
    def __init__(self, bridge_len):
        self.nodes_nums =    []
        self.nodes      =    OrderedDict()   # {node_num: node}

        self.units      =    OrderedDict()   # {unit_num: unit}
        self.units_nums =    []


        self._length    =    bridge_len            # 长度/米
        self.h          =    None
        self._K         =    None
        self._reduced_K =    None
        self._n_nodes   =    None
        self._E         =    None
        self.bottom_chord_length = None

        self.load       =    None

        self.supports   =    {}
        self.supports_nodes_nums   =    None     # 支座节点编号


        self._checking   =   False           # 检算状态


        self.bottom_chord_nodes_nums = []    # 下弦杆节点编号

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
        ''' 读入节点数据 '''
        
        # 读入节点坐标数据
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

        # 读入下弦杆节点数据
        file_name = 'bottom_chord_nodes_nums_' + str(self.length) + '.txt'
        file_path = os.path.join(path, 'nodes', file_name)
        bottom_chord_nodes_nums = []
        with open(file_path, 'r') as bottom_chord_nodes_nums_file:
            for line in bottom_chord_nodes_nums_file.readlines():
                bottom_chord_nodes_nums += [int(string) for string in line.strip().split()]

        self.bottom_chord_nodes_nums = sorted(bottom_chord_nodes_nums)


                
    
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
        # file_name = 'section_params_' + str(self.length) + '.txt'
        # file_path = os.path.join(path, 'units', file_name)
        # with open(file_path, 'r') as section_params_file:
        #     for line in section_params_file.readlines():
        #         # 单元编号 腹板宽度 翼缘厚度 腹板厚度 翼缘宽度
        #         section_params = line.strip().split()
        #         self.units[int(section_params[0])].section_params = [float(string) for string in section_params[1:]]

        
        # 读入杆件类型
        # 0 - 上弦杆
        # 1 - 下弦杆
        # 2 - 端斜梁、端立杆、中间支点处立柱
        # 3 - 竖杆 (包含2中端立杆)
        file_name = 'units_types_data_' + str(self.length) + '.txt'
        file_path = os.path.join(path, 'units', file_name)
        with open(file_path, 'r', encoding='utf-8') as units_types_data_file:
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
                    self.bottom_chord_nodes_indices = list(range(len(self.bottom_chord_nodes_nums)))
                elif type_num == 2:
                    self.side_units_nums = curr_units_nums
                elif type_num == 3:
                    self.vertical_units = curr_units_nums

                # for unit_num in curr_units_nums:
                #     self.units[int(unit_num)].type_ = type_num

            other_units_nums = set(self.units_nums) - set(not_other_units_nums)
            # for unit_num in other_units_nums:
            #     self.units[unit_num].type_ = 3
            self.other_units_nums = other_units_nums


    def load_section_params(self, path):
        file_name = 'section_params_' + str(self.length) + '.txt'
        file_path = os.path.join(path, 'units', file_name)
        with open(file_path, 'r') as section_params_file:
            for line in section_params_file.readlines():
                # 单元编号 腹板宽度 翼缘厚度 腹板厚度 翼缘宽度
                section_params = line.strip().split()
                self.units[int(section_params[0])].section_params = [float(string) for string in section_params[1:]]

        
                

                
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
        self.P = kargs['P']

        self._E = kargs['E']
        for unit_num, unit in self.units.items():
            unit._E = self._E
        
        self.h = kargs['h']
        for unit_num, unit in self.units.items():
            unit.h = self.h
               
        # self.bottom_chord_nodes_nums = kargs['bottom_chord_nodes_nums']
        # self.bottom_chord_nodes_indices = list(range(len(self.bottom_chord_nodes_nums)))
        self.bottom_chord_length = kargs['bottom_chord_length']
        self.load = kargs['load']

                  



#################### 属性 ########################

    def set_supports(self, supports):
        '''
        设置支座
            参数 - 节点编号列表    
        '''
        for support in sorted(supports, key=lambda support: support.node_num):
            self.supports[support.node_num] = support

        bc_nodes_nums = self.bottom_chord_nodes_nums
        # 在下弦杆节点中对应索引，便于计算节点位移中使用
        supports_nodes_nums = [support.node_num for support in self.supports.values()]
        self.supports_nodes_indices = [
            bc_nodes_nums.index(support_node_num) for support_node_num in supports_nodes_nums
        ]


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
            K = seat_kij(K, unit, unit.kij)
            
        return reshape_K(K, self.n_nodes)

    @property
    def K(self):
        '''总体刚度矩阵'''
        if self._K is None:
            self._K = self.get_K()
        
        return self._K
    
    @property
    def reduced_K(self):
        if self._reduced_K is None:
            self._reduced_K = self.reduce_K(self.K)

        return self._reduced_K


    def reduce_K(self, K):
        '''
        缩减刚度矩阵
            由于约束，去掉下弦杆第1个节点横向、竖向位移对应的行列，
            以及最后1个和支座节点竖向位移对应的行列
        '''
        
        # 64m无支座: 去掉1行1列，2行2列，32行32列
        # return np.delete(np.delete(K, [0, 1, 31], axis=0), [0, 1, 31], axis=1)
        
        # 160m加支座: 去掉1行1列，2行2列，42行42列，80行80列
        # return np.delete(np.delete(K, [0, 1, 41, 79], axis=0), [0, 1, 41, 79], axis=1)
        
        # target_indices = [1]\
        #                + [2 * node_num - 2 for node_num in self.supports_nodes_nums]
        #                + [2 * node_num - 1 for node_num in self.supports_nodes_nums]\
        #                + [2 * self.bottom_chord_nodes_nums[-1] - 1] 

        target_indices = []
        for support in self.supports.values():
            if support.h:
                target_indices.append(2 * support.node_num - 2)
            if support.v:
                target_indices.append(2 * support.node_num - 1)
        
        if self.length == 64:
            assert target_indices == [0, 1, 31]
        elif self.length == 160:
            assert target_indices == [1, 40, 41, 79]

        return np.delete(np.delete(K, target_indices, axis=0), target_indices, axis=1)
    

    ## 调整截面参数后更新
    def update_units_A_m(self):
        for unit in self.units.values():
            unit._A_m = unit.get_A_m()
        print("所有单元 [毛面积] 已重新计算.")


    def update_units_kij(self):
        for unit in self.units.values():
            unit._kij = unit.get_kij()
        print("所有单元 [单元刚度矩阵] 已重新计算.")


    def update_K(self):
        self._K = self.get_K()
        print("[总体刚度矩阵] 已重新计算.")


    def update_reduced_K(self):
        self._reduced_K = reduce_K(self.K, self.length)
        print("[缩减刚度矩阵] 已重新计算.")
    


######################## 计算节点竖向位移 ####################

    def save_nodes_vdisps_moment(self, nodes_vdisps_moment):
        '''将[某一]时刻[所有]节点的竖向位移保存到各个节点的竖向位移向量中
        随桥结构变化'''
        
        # 子类中实现
        #raise NotImplementedError
        bc_nodes_nums = self.bottom_chord_nodes_nums
        s_nodes_nums = self.supports.keys()
        supports = self.supports
        
        # 由于去掉某行列导致的索引偏差
        # 用于保存节点竖向位移
        # self.nodes_vdisps_acc_offsets 用于计算时使用
        acc_offsets_for_saving = 0 
        for node_num in self.nodes.keys():
            # 第1、21、80个节点竖向位移不在位移向量D中，为0
            if node_num in s_nodes_nums:
                support = supports[node_num]
                if support.v:
                    self.nodes[node_num].vdisps.append(0.)
                    acc_offsets_for_saving += 1

            else:
                # 160m 加支座为例
                # v2, v3, v4, ..., v20 <= 1, 3, 5,  ...,  37
                # (v21), v22, v23, v24 ..., v79, v(80) <= (), 40, 42, 44  ...,  154, ()
                # 不依赖于编号方式
                self.nodes[node_num].vdisps.append(
                    float(nodes_vdisps_moment[2 * node_num - 3 - acc_offsets_for_saving])
                )
                
    
    
    def get_nodes_vdisps_moment_on_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点上][所有]节点的竖向位移
        随桥结构变化'''
        
        # 子类中实现
        #raise NotImplementedError
        
        bc_nodes_nums = self.bottom_chord_nodes_nums
        
        s_nodes_indices = self.supports_nodes_indices
        supports = self.supports
        curr_node_index = int(point // 8)

        acc_offsets = self.nodes_vdisps_acc_offsets
        # 第1个、最后1个和支座对应索引
        if curr_node_index in s_nodes_indices:
            support = supports[bc_nodes_nums[curr_node_index]]
            if support.h and support.v:
                acc_offsets += 2
                D = np.zeros((self.reduced_K.shape[0], 1))
            
            elif not support.h and support.v:
                acc_offsets += 1
                D = np.zeros((self.reduced_K.shape[0], 1))

            elif support.h and not support.v:
                acc_offsets += 1
                F = np.zeros((self.reduced_K.shape[0], 1))
                F[4 * (curr_node_index + 1) - 5 - acc_offsets] = - self.P
                D = np.matmul(np.linalg.inv(self.reduced_K), F)


                
            
        else:
            F = np.zeros((self.reduced_K.shape[0], 1))
            # y3, y5, y7, y11, ..., y19  <= 3, 7, 11, 15, ..., 35
            # y23, y25, y27, ..., y79  <=

            # 计算出当前节点之前支座对应的节点
            # curr_node_with_s_nodes_indices = [curr_node_index] + s_nodes_indices
            # curr_node_index_with_s_nodes = sorted(curr_node_with_s_nodes_indices).index(curr_node_index)
            # passed_s_nodes_indices = curr_node_with_s_nodes_indices[:curr_node_index_with_s_nodes + 1]
            # passed_s_nodes_nums = [bc_nodes_nums[passed_s_node_index] for passed_s_node_index in passed_s_nodes_indices]

            # acc_offsets = self.nodes_vdisps_acc_offsets
            # for passed_s_node_num in passed_s_nodes_nums:
            #     support = supports[passed_s_node_num]
            #     if suport.v:
            #         acc_offsets += 1
            #     if support.h:
            #         acc_offsets += 1
            
            
            F[4 * (curr_node_index + 1) - 5 - acc_offsets] = - self.P
                
            D = np.matmul(np.linalg.inv(self.reduced_K), F)

        self.nodes_vdisps_acc_offsets = acc_offsets
            
        return D
    
    
    def get_nodes_vdisps_moment_between_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点间][所有]节点的竖向位移
        随桥结构变化'''
        
        # 子类中实现
        #raise NotImplementedError
        #bc_nodes_indices = self.bottom_chord_nodes_indices
        bc_nodes_nums = self.bottom_chord_nodes_nums
        s_nodes_indices = self.supports_nodes_indices
        supports = self.supports
        bc_len = self.bottom_chord_length
        
        prev_node_index = int(point // bc_len)
        next_node_index = prev_node_index + 1

        prev_node_offset = point - bc_len * prev_node_index   # 距离前一个节点的位移偏移量, 每个节点间距离8m
        #print(prev_node_index, next_node_index, prev_node_offset)

        next_node_force = - prev_node_offset / bc_len  # 由杠杆原理得出作用在下一个节点的力
        prev_node_force = - 1 - next_node_force

        # 以64m桥为例
        # 在节点间(1, 3)时不需要算分摊到节点1上的力，对应索引(0, 1)
        # 在节点间(15, 16)时不需要算分摊到节点16上的力，对应索引(7, 8)
        F = np.zeros((self.reduced_K.shape[0], 1))
        # prev_index_offset = sorted([prev_node_index] + s_nodes_indices).index(prev_node_index)
        # next_index_offset = sorted([next_node_index] + s_nodes_indices).index(next_node_index)

        # 计算出当前节点之前支座对应的节点
        # prev_node_with_s_nodes_indices = [prev_node_index] + s_nodes_indices
        # prev_node_index_with_s_nodes = sorted(prev_node_with_s_nodes_indices).index(prev_node_index)
        # passed_s_nodes_indices = prev_node_with_s_nodes_indices[:prev_node_index_with_s_nodes + 1]
        # passed_s_nodes_nums = [bc_nodes_nums[passed_s_node_index] for passed_s_node_index in passed_s_nodes_indices]

        # acc_offsets = self.nodes_vdisps_acc_offsets
        # for passed_s_node_num in passed_s_nodes_nums:
        #     support = supports[passed_s_node_num]
        #     if suport.v:
        #         acc_offsets += 1
        #     if support.h:
        #         acc_offsets += 1

        acc_offsets = self.nodes_vdisps_acc_offsets
        if prev_node_index in s_nodes_indices:
            # 上一个节点是支座
            support = supports[bc_nodes_nums[prev_node_index]]
            if not support.v:
                F[4 * (next_node_index + 1) - 5 - acc_offsets] = next_node_force

        elif next_node_index in s_nodes_indices:
            support = supports[bc_nodes_nums[next_node_index]]
            if not support.v:
                F[4 * (prev_node_index + 1) - 5 - acc_offsets] = prev_node_force

        else:
            F[4 * (prev_node_index + 1) - 5 - acc_offsets] = prev_node_force
            F[4 * (next_node_index + 1) - 5 - acc_offsets] = next_node_force

        D = np.matmul(np.linalg.inv(self.reduced_K), F)

        return D


        # if prev_node_index == s_nodes_indices[0]:
        #     F[4 * (next_node_index + 1) - 5 - 0] = next_node_force

        # elif next_node_index == s_nodes_indices[-1]:
        #     F[4 * (prev_node_index + 1) - 5 - prev_index_offset] = prev_node_force

        # else:
        #     if next_node_index in s_nodes_indices:
        #         F[4 * (prev_node_index + 1) - 5 - prev_index_offset] = prev_node_force

        #     elif prev_node_index in s_nodes_indices:
        #         F[4 * (next_node_index + 1) - 5 - next_index_offset] = next_node_force

        #     else:
        #         # prev_node_offset == next_node_offset
        #         F[4 * (prev_node_index + 1) - 5 - prev_index_offset] = prev_node_force
        #         F[4 * (next_node_index + 1) - 5 - prev_index_offset] = next_node_force

        
        # D = np.matmul(np.linalg.inv(self.reduced_K), F)
        # return D
    
    
    
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
        #if int(point) == point and point in bc_nodes_corr_force_disps:
        if point in bc_nodes_corr_force_disps:
            D = self.get_nodes_vdisps_moment_on_nodes(point)

        # 力作用在节点间
        else:
            D = self.get_nodes_vdisps_moment_between_nodes(point)

        return D
    
    
    def nodes_vdisps_init(self):
        '''将节点竖向位移清空'''
        for node in self.nodes.values():
            node.vdisps = []
    
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
        self.nodes_vdisps_acc_offsets = 0
        for point in bc_axis:
            nodes_vdisps_moment = self.get_nodes_vdisps_moment(point)
            self.save_nodes_vdisps_moment(nodes_vdisps_moment)

        for node in self.nodes.values():
            node.vdisps = np.array(node.vdisps)
     
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
            unit.axial_forces = []

            
    def save_units_axial_forces_moment(self, units_axial_forces_moment):
        for unit, one_unit_axial_force_moment in zip(self.units.values(), units_axial_forces_moment):
            unit.axial_forces.append(one_unit_axial_force_moment)

            
    def get_one_unit_axial_force_moment(self, unit, nodes_vdisps_moment):
        '''计算某一时刻某一杆件单元的轴力
        随桥结构变化'''
        
        #raise NotImplementedError
        
        def get_u_and_v(node_num, nodes_vdisps_moment):
            '''
            index      0    1   2   3      36   37   38 (   )   39   40   41  42  43  44 ...   155
            (u1) (v1) [u2  v2  u3  v3 ... u20  v20  u21 (v21)  u22  v22  u23 v23 u24 v24 ...   u80] (v80)
            '''
            
            D = nodes_vdisps_moment
            bc_nodes_nums = self.bottom_chord_nodes_nums
            s_nodes_nums = self.supports_nodes_nums
            supports = self.supports
            acc_offsets = self.nodes_vdisps_acc_offsets

            if node_num in s_nodes_nums:
                support = supports[node_num]
                u = float(D[2 * (node_num - 1) - 2 - acc_offsets]) if support.h else 0
                v = float(D[2 * (node_num - 1) - 1 - acc_offsets]) if support.v else 0

            else:
                u = float(D[2 * (node_num - 1) - 2 - acc_offsets])
                v = float(D[2 * (node_num - 1) - 1 - acc_offsets])
            # assert bc_nodes_nums[0] <= node_num <= bc_nodes_nums[-1]  # 1 <= node_num <= 80
            # 第1个节点横向和竖向位移都不在位移向量D中，且为0
            # if node_num == bc_nodes_nums[0]:  
            #     u, v = 0, 0 
            
            # # 中间节点21横向位移为0，竖向位移不在位移向量D中，且为0
            # #  u21, (v21), u22, v22
            # #   38,    (),  39,  40
            # elif node_num in s_nodes_nums:      
            #     u, v = 0, 0                           
            
            # # 最后1个节点横向位移为位移向量D最后一个元素，竖向位移不在位移向量D中，且为0
            # elif node_num == bc_nodes_nums[-1]:
            #     u, v = float(D[-1]), 0

            # else:
            #     index_offset = sorted([node_num] + s_nodes_nums).index(node_num)
                
            #     # print(node_num, index_offset)
            #     u = float(D[2 * (node_num - 1) - 2 - index_offset])
            #     v = float(D[2 * (node_num - 1) - 1 - index_offset])
            
            return u, v
        

        i, j = unit.node_i.num, unit.node_j.num
        ui, vi = get_u_and_v(i, nodes_vdisps_moment)
        uj, vj = get_u_and_v(j, nodes_vdisps_moment) 

        kij = unit.kij
        a = unit.alpha
        dij = np.array([[ui], [vi], [uj], [vj]])

        T = np.array(
            [[cos(a), sin(a), 0, 0],
             [-sin(a), cos(a), 0, 0],
             [0, 0, cos(a), sin(a)],
             [0, 0, -sin(a), cos(a)]])
        
        Fij = np.matmul(T, np.matmul(kij, dij))
        f = sqrt(Fij[0]**2 + Fij[1]**2)           # 轴力大小
        
        #shaft_units = list(range(3, 75 + 1, 4)) # 竖杆
        if unit.num in self.vertical_units:    # 竖杆
            f = f if Fij[0] > 0 else -f
        else:
            f = -f if Fij[0] > 0 else f

        return f
    
    
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
        self.nodes_vdisps_acc_offsets = 0
        for point in bc_axis:
            units_axial_forces_moment = self.get_units_axial_forces_moment(point)
            self.save_units_axial_forces_moment(units_axial_forces_moment)
    
        print('所有杆件单元轴力已计算完毕.')

        
        for unit in self.units.values():
            unit.axial_forces = np.array(unit.axial_forces)

        # 判断零杆
        is_zero = None
        for unit in self.units.values():
            unit_axial_forces = unit.axial_forces
            if (np.around(unit_axial_forces, decimals=8) == 0).all():
                is_zero = True
            else:
                is_zero = False

            unit.is_zero = is_zero

        print('所有零杆已标识.')
    
    
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

    

    

##################### 计算最不利位移 / 荷载 #####################
    
    
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

            pos_max = np.max((pos_max, pos_sum))
            neg_min = np.min((neg_min, neg_sum))

            if int(i - (self.length / 0.1 + 1)) == -load.shape[0]:
                break

            i -= 1

        return pos_max, neg_min



    def get_worst_cases_load(self):
        '''计算所有杆件单元最不利荷载'''
        for unit in self.units.values():
            unit_axial_forces = unit.axial_forces
            load = self.load.load
            #pos_max, neg_min = self.get_unit_worst_cases_load(unit_axial_forces, load, pos_max=-np.inf, neg_min=np.inf)
            pos_max, neg_min = self.get_worst_cases(unit_axial_forces, load)     
            unit.max_forces = (pos_max, neg_min)
            
        print('所有杆件单元最不利荷载已计算完毕.')



    def get_worst_cases_disps(self):
        '''计算所有杆件单元最不利位移'''
        for node in self.nodes.values():
            node_vdisps = node.vdisps
            load = self.load.load
            #pos_max, neg_min = self.get_unit_worst_cases_load(unit_axial_forces, load, pos_max=-np.inf, neg_min=np.inf)
            pos_max, neg_min = self.get_worst_cases(node_vdisps, load)     
            node.max_vdisps = (pos_max, neg_min)
            
        print('所有杆件单元最不利位移已计算完毕.')


    ################ 检算 #######################

    def check(self, checking=True):
        '''打开或关闭检算状态'''
        self._checking = checking
        for unit in self.units.values():
            unit._checking = checking



    def set_section_params(self, units_nums_group, section_params):
        '''
        设置某一组杆件单元的截面参数
            参数 - 杆件单元编号列表
        '''

        for unit_num in units_nums_group:
            unit = self.units[unit_num]
            unit.section_params = section_params

        #print('当前修改参数杆件组\n', units_nums_group)



        
    def update(self):
        '''更新与横截面参数有依赖关系的量'''

        self.update_units_kij()
        self.update_units_A_m()
        self.update_K()
        self.update_reduced_K()

        self.get_nodes_vdisps()
        self.get_units_axial_forces()
        self.get_worst_cases_load()
        self.get_worst_cases_disps()

    
    def get_units_clusters(self, target_units_nums, n_clusters=7):
        '''根据最大内力N聚类'''

    
        target_units_N = [list(self.units[unit_num].N) for unit_num in target_units_nums]

        kmeans = KMeans(n_clusters, max_iter=300, random_state=0).fit_predict(np.array(target_units_N))

        units_clusters = OrderedDict({label: [] for label in range(n_clusters)})
        for (unit_num, label) in zip(target_units_nums, kmeans):
            units_clusters[label].append(unit_num)

        self.units_clusters = units_clusters
        return units_clusters


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