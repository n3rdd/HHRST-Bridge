from math import sin, cos, sqrt
import numpy as np

from bridge_base import Bridge


######################################
#                                    #
#               64m 桥子类            #
#                                    #
######################################

class Bridge_64(Bridge):
    '''
    64米钢桥类（无支座）
    '''
    def __init__(self):
        super(Bridge_64, self).__init__()
        self._length = 64
    
    
################### 计算节点竖向位移 #####################

    def save_nodes_vdisps_moment(self, nodes_vdisps_moment):
        '''将[某一]时刻[所有]节点的竖向位移保存到各个节点的竖向位移向量中'''
        
        bc_nodes_nums = self.bottom_chord_nodes_nums
        #bc_nodes_indices = self.bottom_chord_nodes_indices
        
        for node_num in self.nodes.keys():
            # 第1和16个节点竖向位移不在位移向量D中，为0
            if node_num in [bc_nodes_nums[0], bc_nodes_nums[-1]]:   
                self.nodes[node_num].vdisps.append(0.)
            else:
                # v2, v3, v4, ..., v15 <= 1, 3, 5,  ...,  27
                self.nodes[node_num].vdisps.append(float(nodes_vdisps_moment[2 * (node_num - 1) - 1]))
        
    
    def get_nodes_vdisps_moment_on_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点上][所有]节点的竖向位移'''
        
        bc_nodes_indices = self.bottom_chord_nodes_indices
        current_node_index = int(point // 8)
        if current_node_index in [bc_nodes_indices[0], bc_nodes_indices[-1]]:  # 第1个和最后1个
            D = np.zeros((self.reduced_K.shape[0], 1))             # 取决于reduced_K形状 (29, 29)
        else:
            F = np.zeros((self.reduced_K.shape[0], 1))
            F[4 * (current_node_index + 1) - 5] = - self.P          # y3, y5, y7, y11, ..., y15  <= 3, 7, 11, 15, ..., 27
            D = np.matmul(np.linalg.inv(self.reduced_K), F)
            
        return D
    
    
    def get_nodes_vdisps_moment_between_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点间][所有]节点的竖向位移'''
        
        bc_nodes_indices = self.bottom_chord_nodes_indices
        
        F = np.zeros((self.reduced_K.shape[0], 1))
        prev_node_index = int(point // 8)
        next_node_index = prev_node_index + 1

        prev_node_offset = point - 8 * prev_node_index   # 距离前一个节点的位移偏移量, 每个节点间距离8m
        #print(prev_node_index, next_node_index, prev_node_offset)

        next_node_force = - prev_node_offset / 8  # 由杠杆原理得出作用在下一个节点的力
        prev_node_force = - 1 - next_node_force

        # 以64m桥为例
        # 在节点间(1, 3)时不需要算分摊到节点1上的力，对应索引(0, 1)
        # 在节点间(15, 16)时不需要算分摊到节点16上的力，对应索引(7, 8)
        if prev_node_index != bc_nodes_indices[0]:        
            F[4 * (prev_node_index + 1) - 5] = prev_node_force 
        if prev_node_index != bc_nodes_indices[-2]:
            F[4 * (next_node_index + 1) - 5] = next_node_force

        D = np.matmul(np.linalg.inv(self.reduced_K), F)
        return D


    

################# 计算杆件轴力 ###################  
    def get_one_unit_axial_force_moment(self, unit, nodes_vdisps_moment):
        '''计算某一时刻某一杆件单元的轴力'''
        
        i, j = unit.node_i.num, unit.node_j.num

        def get_u_and_v(node_num, nodes_vdisps_moment):
            '''
            index      0    1   2   3      26   27   28 
            (u1) (v1) [u2  v2  u3  v3 ... u15  v15  u16] (v16) 
            '''
            
            D = nodes_vdisps_moment
            bc_nodes_nums = self.bottom_chord_nodes_nums
            
            assert bc_nodes_nums[0] <= node_num <= bc_nodes_nums[-1]  
            # 第1个节点横向和竖向位移都不在位移向量D中，且为0
            if node_num == bc_nodes_nums[0]:  
                u, v = 0, 0
            
            elif bc_nodes_nums[0] < node_num < bc_nodes_nums[-1]:
                u = float(D[2 * (node_num - 1) - 2])  
                v = float(D[2 * (node_num - 1) - 1])  
            
            
            # 最后1个节点横向位移为位移向量D最后一个元素，竖向位移不在位移向量D中，且为0
            elif node_num == bc_nodes_nums[-1]:
                u, v = float(D[-1]), 0
            
            return u, v
        
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
        
        shaft_units = [3, 7, 11, 15, 19, 23, 27]
        if unit.num in shaft_units:    # 竖杆
            f = f if Fij[0] > 0 else -f
        else:
            f = -f if Fij[0] > 0 else f

        
        one_unit_axial_force_moment = f
        return one_unit_axial_force_moment




    
    
######################################
#                                    #
#               160m 桥子类           #
#                                    #
######################################

class Bridge_160(Bridge):
    '''
    160米钢桥类（80米 + 支座 + 80米）
    '''
    def __init__(self):
        super(Bridge_160, self).__init__()
        self._length = 160
        
        
#################### 计算节点竖向位移 ################
    def save_nodes_vdisps_moment(self, nodes_vdisps_moment):
        '''将[某一]时刻[所有]节点的竖向位移保存到各个节点的竖向位移向量中'''
        
        bc_nodes_nums = self.bottom_chord_nodes_nums
        bc_middle_node_num = bc_nodes_nums[len(bc_nodes_nums) // 2]  # 第21个节点
        
        for node_num in self.nodes.keys():
            # 第1、21、80个节点竖向位移不在位移向量D中，为0
            
            if node_num in [bc_nodes_nums[0], bc_middle_node_num, bc_nodes_nums[-1]]:   
                self.nodes[node_num].vdisps.append(0.)
            else:
                # v2, v3, v4, ..., v20 <= 1, 3, 5,  ...,  37
                if node_num < bc_middle_node_num:
                    self.nodes[node_num].vdisps.append(float(nodes_vdisps_moment[2 * node_num - 3]))
                
                # (v21), v22, v23, v24 ..., v79, v(80) <= (), 40, 42, 44  ...,  154, ()
                else:
                    self.nodes[node_num].vdisps.append(float(nodes_vdisps_moment[2 * node_num - 4]))
    
    
    def get_nodes_vdisps_moment_on_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点上][所有]节点的竖向位移'''
        '''
        bc_indices  0   1   2   3   4   5   6   7   8   9  10 ...   19   20 
        bc_nodes    1   3   5   7   9  11  13  15  17  19  21 ...   39   40
        bc_axis     0   8  16  24  32  40  48  56  64  72  80 ...  152  160
        '''
        
        
        bc_middle_node_index = len(self.bottom_chord_nodes_nums) // 2  # 第21个节点
        
        bc_nodes_indices = self.bottom_chord_nodes_indices
        current_node_index = int(point // 8)
        if current_node_index in [bc_nodes_indices[0], bc_middle_node_index, bc_nodes_indices[-1]]:  # 第1、21和最后1个
            D = np.zeros((self.reduced_K.shape[0], 1))             # 取决于reduced_K形状 (76, 76)
        else:
            F = np.zeros((self.reduced_K.shape[0], 1))
            if current_node_index < bc_middle_node_index:
                # y3, y5, y7, y11, ..., y19  <= 3, 7, 11, 15, ..., 35
                F[4 * (current_node_index + 1) - 5] = - self.P
            else:
                # y23, y25, y27, ..., y79  <= 
                F[4 * (current_node_index + 1) - 6] = - self.P
                
            D = np.matmul(np.linalg.inv(self.reduced_K), F)
            
        return D
    
    
    def get_nodes_vdisps_moment_between_nodes(self, point):
        '''计算力移动过程中，[某一]时刻力作用在[节点间][所有]节点的竖向位移'''
        
        bc_nodes_indices = self.bottom_chord_nodes_indices
        bc_middle_node_index = len(self.bottom_chord_nodes_nums) // 2  # 第21个节点
        
        
        prev_node_index = int(point // 8)
        next_node_index = prev_node_index + 1

        prev_node_offset = point - 8 * prev_node_index   # 距离前一个节点的位移偏移量, 每个节点间距离8m
        #print(prev_node_index, next_node_index, prev_node_offset)

        next_node_force = - prev_node_offset / 8  # 由杠杆原理得出作用在下一个节点的力
        prev_node_force = - 1 - next_node_force

        # 以64m桥为例
        # 在节点间(1, 3)时不需要算分摊到节点1上的力，对应索引(0, 1)
        # 在节点间(15, 16)时不需要算分摊到节点16上的力，对应索引(7, 8)
        F = np.zeros((self.reduced_K.shape[0], 1))
        
        # ... 17 v 19   21   23 ...
        if next_node_index <= bc_middle_node_index - 1:
            if prev_node_index == bc_nodes_indices[0]:        
                F[4 * (next_node_index + 1) - 5] = next_node_force
            else:
                F[4 * (prev_node_index + 1) - 5] = prev_node_force
                F[4 * (next_node_index + 1) - 5] = next_node_force
            #else: F = 0
        
        # ... 17  19   21  v  23 ...
        elif prev_node_index >= bc_middle_node_index + 1:
            if next_node_index == bc_nodes_indices[-1]:
                F[4 * (prev_node_index + 1) - 6] = prev_node_force    
            else:
                F[4 * (prev_node_index + 1) - 6] = prev_node_force
                F[4 * (next_node_index + 1) - 6] = next_node_force

        # ... 17  19  v  21    23 ...
        # ... 17  19     21 v  23 ...
        else:
            if next_node_index == 21:
                F[4 * (prev_node_index + 1) - 5] = prev_node_force
            elif prev_node_index == 21:
                F[4 * (next_node_index + 1) - 6] = next_node_force
        
        D = np.matmul(np.linalg.inv(self.reduced_K), F)
        return D


    

    
################# 计算杆件单元轴力 ################    
    def get_one_unit_axial_force_moment(self, unit, nodes_vdisps_moment):
        '''计算某一时刻某一杆件单元的轴力'''
        
        i, j = unit.node_i.num, unit.node_j.num
        
                
        def get_u_and_v(node_num, nodes_vdisps_moment):
            '''
            index      0    1   2   3      36   37   38 (   )   39   40   41  42  43  44 ...   155
            (u1) (v1) [u2  v2  u3  v3 ... u20  v20  u21 (v21)  u22  v22  u23 v23 u24 v24 ...   u80] (v80)
            '''
            
            D = nodes_vdisps_moment
            bc_nodes_nums = self.bottom_chord_nodes_nums
            bc_middle_node_num = bc_nodes_nums[len(bc_nodes_nums) // 2]  # 第21个节点
            
            assert bc_nodes_nums[0] <= node_num <= bc_nodes_nums[-1]  # 1 <= node_num <= 80
            # 第1个节点横向和竖向位移都不在位移向量D中，且为0
            if node_num == bc_nodes_nums[0]:  
                u, v = 0, 0
            
            elif bc_nodes_nums[0] < node_num < bc_middle_node_num:
                u = float(D[2 * (node_num - 1) - 2])  
                v = float(D[2 * (node_num - 1) - 1])  
            
            # 中间节点21横向位移为0，竖向位移不在位移向量D中，且为0
            #  u21, (v21), u22, v22
            #   38,    (),  39,  40
            elif node_num == bc_middle_node_num:      
                u, v = 0, 0                           
            
            elif bc_middle_node_num < node_num < bc_nodes_nums[-1]:
                u = float(D[2 * (node_num - 1) - 3])
                v = float(D[2 * (node_num - 1) - 2])  
            
            # 最后1个节点横向位移为位移向量D最后一个元素，竖向位移不在位移向量D中，且为0
            elif node_num == bc_nodes_nums[-1]:
                u, v = float(D[-1]), 0
            
            return u, v
        
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
        
        shaft_units = list(range(3, 75 + 1, 4)) # 竖杆
        if unit.num in shaft_units:    # 竖杆
            f = f if Fij[0] > 0 else -f
        else:
            f = -f if Fij[0] > 0 else f

        
        one_unit_axial_force_moment = f
        return one_unit_axial_force_moment