#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
2/23  0.0.1  完成基本计算流程 
2/24  0.0.2  可视化中间结果 
3/17  0.1.0  结构变化=>64m 
3/18  1.0.0  代码重构 
3/25  1.1.0  更改力每次移动为0.1m 
3/26  1.1.1  实现zk活载 
4/2   2.0.0  代码重构 
4/7   2.1.0  支持160m桥 
4/8   2.2.0  增加64m, 160m桥子类 
4/11  2.2.1  完成疲劳、强度检算 
4/12  2.2.2  完成刚度、整体局部稳定性检算 
4/16  2.2.3  完成截面调节模块 
'''
'''
TODO:
- get_units_axial_forces_moment需检查和self.units顺序是否对应
- get_one_unit_axial_force_moment中get_u_and_v中node_num应该与units.keys()比较而不是bc_nodes_nums
- setter => 自动计算
- reduce_K 修改
- 添加Checker 类
'''


# In[ ]:





# ### 导入模块

# In[49]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from bridge_base import ZkLoad
from bridge import Bridge_64, Bridge_160


# ### 输入参数

# In[50]:


# 常量
E = 21 * 10**10           # 杨氏模量 (Pa)
P = 1                     # 外力    (kN)
h = 2.52                  # 恒载    (kN)


bottom_chord_length = 8.   # 下弦杆长度/下节点间距 (m)

path = './data/'
bridge_len = 160

if bridge_len == 64:
    bridge = Bridge_64()
    bottom_chord_nodes_nums = [1, 3, 5, 7, 9, 11, 13, 15, 16]  # 下弦杆节点编号

elif bridge_len == 160:
    bridge = Bridge_160()
    # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 40]
    bottom_chord_nodes_nums = list(range(1, 40, 2)) + [40]


# In[3]:


'''
步长 = 0.1m

       6.4      0             200    
     均布荷载   填充          集中荷载                     均布荷载
{| | | | | |} . . . {#  .  #  .   #  .  #}  . . .  {| | | | | |}
    桥长度       0.8    1.6     1.6   1.6       0.8   桥长度      
'''




# In[52]:


zk_load = ZkLoad(bridge.length)


# ### 载入数据

# In[75]:


bridge.load_data(path)
bridge.load_params(E=E, P=P, h=h, 
                   bottom_chord_nodes_nums=bottom_chord_nodes_nums,
                   bottom_chord_length=bottom_chord_length, 
                   load=zk_load)







units_group = bridge.units_nums
# 梁截面数据([腹板宽度 翼缘厚度 腹板厚度 翼缘宽度])
#             b2       t1      t2     b1
print('检算=> 初始化截面参数')
b1, t1, b2, t2 = 0.215, 0.012, 0.436, 0.010
beam_section_data = [b2, t1, t2, b1]
print(beam_section_data)
bridge.check()
bridge.set_section_params(units_group, beam_section_data)
bridge.update()


# In[25]:


import numpy as np
from sklearn.cluster import KMeans
from collections import OrderedDict
import matplotlib.pyplot as plt

target_units_nums = list(range(1, 40))
target_units_max_forces = [list(bridge.units[unit_num].max_force) for unit_num in target_units_nums]


# In[28]:


X = np.array(target_units_max_forces)
n_clusters = 7
kmeans = KMeans(n_clusters, max_iter=300, random_state=0).fit_predict(X)



# In[32]:


clusters = OrderedDict({label: [] for label in range(n_clusters)})
for unit_num, unit_max_forces, label in zip(target_units_nums, target_units_max_forces, kmeans):
    clusters[label].append(unit_num)
# for label, cluster in clusters.items():
#     print(label)
#     for unit_num in cluster:
#         print(unit_num, bridge.units[unit_num].max_force)
#     print()


# In[33]:


clusters


# In[ ]:


clusters[2] += [43, 47, 51, 55, 59, 63, 67, 71, 75]
clusters[1] += [42]
clusters[5] += [41]
clusters[6] += [44, 48, 53, 61, 65, 73, 74, 77]
clusters[0] += [40, 45, 50, 54, 58, 62, 66, 70]
clusters[3] += [52, 56, 60, 64]
clusters[4] += [46, 49, 57, 68, 69, 72, 76]
for i, cluster in clusters.items():
    print(i, cluster)
# In[65]:


bh_cases = []
B = [0.46, 0,60, 0.72]
H = [0.44, 0.60, 0.76]
c2_h = 0.44
for b in B:
    for c0_h in H:
        for c1_h in H:
            for c3_h in H:
                for c4_h in H:
                    for c5_h in H:
                        for c6_h in H:
                            # 7 类
                            # ((b, h), (b, h), ..., (b, h))
                            bh_cases.append(((b, c0_h), (b, c1_h), (b, c2_h), (b, c3_h), 
                                          (b, c4_h), (b, c5_h), (b, c6_h)))



results_file = open('results.txt', 'w')
section_cases = []
for bh_case in bh_cases:
    c0_bh, c1_bh, c2_bh, c3_bh, c4_bh, c5_bh, c6_bh = bh_case
    c0_b, c0_h = c0_bh
    c1_b, c1_h = c1_bh
    c2_b, c2_h = c2_bh
    c3_b, c3_h = c3_bh
    c4_b, c4_h = c4_bh
    c5_b, c5_h = c5_bh
    c6_b, c6_h = c6_bh
    
    # c0
    for c0_t1 in np.arange(0.01, c0_b / 2, 0.01):
        c0_b2 = c0_b - 2 * c0_t1
        for c0_b1 in np.arange(0.02, c0_h / 2, 0.02):
            c0_t2 = c0_h - 2 * c0_b1
            
            # c1
            for c1_t1 in np.arange(0.01, c1_b / 2, 0.01):
                c1_b2 = c1_b - 2 * c1_t1
                for c1_b1 in np.arange(0.02, c1_h / 2, 0.02):
                    c1_t2 = c1_h - 2 * c1_b1
                    
                    # c2
                    for c2_t1 in np.arange(0.01, c2_b / 2, 0.01):
                        c2_b2 = c2_b - 2 * c2_t1
                        for c2_b1 in np.arange(0.02, c2_h / 2, 0.02):
                            c2_t2 = c2_h - 2 * c2_b1
                            
                            # c3
                            for c3_t1 in np.arange(0.01, c3_b / 2, 0.01):
                                c3_b2 = c3_b - 2 * c3_t1
                                for c3_b1 in np.arange(0.02, c3_h / 2, 0.02):
                                    c3_t2 = c3_h - 2 * c3_b1
                                    
                                    # c4
                                    for c4_t1 in np.arange(0.01, c4_b / 2, 0.01):
                                        c4_b2 = c4_b - 2 * c4_t1
                                        for c4_b1 in np.arange(0.02, c4_h / 2, 0.02):
                                            c4_t2 = c4_h - 2 * c4_b1
                                            
                                            # c5
                                            for c5_t1 in np.arange(0.01, c5_b / 2, 0.01):
                                                c5_b2 = c5_b - 2 * c5_t1
                                                for c5_b1 in np.arange(0.02, c5_h / 2, 0.02):
                                                    c5_t2 = c5_h - 2 * c5_b1
                                                    
                                                    # c6
                                                    for c6_t1 in np.arange(0.01, c6_b / 2, 0.01):
                                                        c6_b2 = c6_b - 2 * c6_t1
                                                        for c6_b1 in np.arange(0.02, c6_h / 2, 0.02):
                                                            c6_t2 = c6_h - 2 * c6_b1
                                                            
                                                            # b2, t1, t2, b1
                                                            

                                                            print((c0_b2, c0_t1, c0_t2, c0_b1)) 
                                                            print((c1_b2, c1_t1, c1_t2, c1_b1))
                                                            print((c2_b2, c2_t1, c2_t2, c2_b1)) 
                                                            print((c3_b2, c3_t1, c3_t2, c3_b1)) 
                                                            print((c4_b2, c4_t1, c4_t2, c4_b1)) 
                                                            print((c5_b2, c5_t1, c5_t2, c5_b1)) 
                                                            print((c6_b2, c6_t1, c6_t2, c6_b1))
                            
                                                            bridge.check()
    
                                                            beam_section_0 = [c0_b2, c0_t1, c0_t2, c0_b1]
                                                            beam_section_1 = [c1_b2, c1_t1, c1_t2, c1_b1]
                                                            beam_section_2 = [c2_b2, c2_t1, c2_t2, c2_b1]
                                                            beam_section_3 = [c3_b2, c3_t1, c3_t2, c3_b1]
                                                            beam_section_4 = [c4_b2, c4_t1, c4_t2, c4_b1]
                                                            beam_section_5 = [c5_b2, c5_t1, c5_t2, c5_b1]
                                                            beam_section_6 = [c6_b2, c6_t1, c6_t2, c6_b1]
                                                            bridge.set_section_params(clusters[0], beam_section_0)
                                                            bridge.set_section_params(clusters[1], beam_section_1)
                                                            bridge.set_section_params(clusters[2], beam_section_2)
                                                            bridge.set_section_params(clusters[3], beam_section_3)
                                                            bridge.set_section_params(clusters[4], beam_section_4)
                                                            bridge.set_section_params(clusters[5], beam_section_5)
                                                            bridge.set_section_params(clusters[6], beam_section_6)
                                                                            
                                                            bridge.update()

                                                            

                                                            # try:
                                                            #     qualified = bridge.stiffness_check()
                                                            
                                                            # except ValueError:
                                                            #     print('# ValueError\n', current_case, file=results_file)
                                                            #     continue
                                                            
                                                            # else:
                                                            #     if qualified:
                                                            #         print('合格')   
                                                            #         print('# Qualified\n', qualified_case, file=results_file)
                                                            #     else:
                                                            #         print('不合格')
                                                            qualified = bridge.stiffness_check()
                                                            if qualified:
                                                                current_case = (
                                                                    (c0_b2, c0_t1, c0_t2, c0_b1), 
                                                                    (c1_b2, c1_t1, c1_t2, c1_b1), 
                                                                    (c2_b2, c2_t1, c2_t2, c2_b1), 
                                                                    (c3_b2, c3_t1, c3_t2, c3_b1), 
                                                                    (c4_b2, c4_t1, c4_t2, c4_b1), 
                                                                    (c5_b2, c5_t1, c5_t2, c5_b1), 
                                                                    (c6_b2, c6_t1, c6_t2, c6_b1)
                                                                )
                                                                print('合格')
                                                                print(current_case, file=results_file)
                                                            else:
                                                                print('不合格')
                                    
                                    
            
results_file.close()





