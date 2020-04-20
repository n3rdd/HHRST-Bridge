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



bridge.load_data(path)
bridge.load_params(E=E, P=P, h=h, 
                   bottom_chord_nodes_nums=bottom_chord_nodes_nums,
                   bottom_chord_length=bottom_chord_length, 
                   load=zk_load)





units_group = bridge.units_nums
# 梁截面数据([腹板宽度 翼缘厚度 腹板厚度 翼缘宽度])
#             b2       t1      t2     b1
print('检算 => 初始化截面参数')
b1, t1, b2, t2 = 0.215, 0.012, 0.436, 0.010
beam_section_data = [b2, t1, t2, b1]
print(beam_section_data)
bridge.check()
bridge.set_section_params(units_group, beam_section_data)
bridge.update()



import numpy as np
from sklearn.cluster import KMeans
from collections import OrderedDict
import matplotlib.pyplot as plt

target_units_nums = list(range(1, 40))
target_units_max_forces = [list(bridge.units[unit_num].max_force) for unit_num in target_units_nums]



X = np.array(target_units_max_forces)
n_clusters = 7
kmeans = KMeans(n_clusters, max_iter=300, random_state=0).fit_predict(X)



clusters = OrderedDict({label: [] for label in range(n_clusters)})
for unit_num, unit_max_forces, label in zip(target_units_nums, target_units_max_forces, kmeans):
    clusters[label].append(unit_num)
for label, cluster in clusters.items():
    print(label)
    for unit_num in cluster:
        print(unit_num, bridge.units[unit_num].max_force)
    print()



clusters[2] += [43, 47, 51, 55, 59, 63, 67, 71, 75]
clusters[1] += [42]
clusters[5] += [41]
clusters[6] += [44, 48, 53, 61, 65, 73, 74, 77]
clusters[0] += [40, 45, 50, 54, 58, 62, 66, 70]
clusters[3] += [52, 56, 60, 64]
clusters[4] += [46, 49, 57, 68, 69, 72, 76]
for i, cluster in clusters.items():
    print(i, cluster)


# 刚度、局部稳定、疲劳
qualified_count = 0
qualified_cases = []
count = 0
results_file = open('stage2_results.txt', 'w')
last_qualified_cases = np.load("qualified_cases.npy")


# 整体稳定、强度
for case in last_qualified_cases:
    bridge.check()
    beam_section_0 = case[0]
    beam_section_1 = case[1]
    beam_section_2 = case[2]
    beam_section_3 = case[3]
    beam_section_4 = case[4]
    beam_section_5 = case[5]
    beam_section_6 = case[6]
    bridge.set_section_params(clusters[0], beam_section_0)
    bridge.set_section_params(clusters[1], beam_section_1)
    bridge.set_section_params(clusters[2], beam_section_2)
    bridge.set_section_params(clusters[3], beam_section_3)
    bridge.set_section_params(clusters[4], beam_section_4)
    bridge.set_section_params(clusters[5], beam_section_5)
    bridge.set_section_params(clusters[6], beam_section_6)           
    bridge.update()

    strength_qualified = bridge.strength_check()
    overall_stability_qualified = bridge.overall_stability_check()
    qualified = np.array((strength_qualified, overall_stability_qualified)).all()

    
    count += 1
    print('%2d|  (强度)%s|  (整体稳定)%s|  %s' % ( count,  
                                            ('合格' if strength_qualified else '不合格'),
                                            ('合格' if overall_stability_qualified else '不合格'), 
                                            ('合格' if qualified else '不合格')))
    print('%2d|  (强度)%s|  (整体稳定)%s|  %s' % ( count,  
                                            ('合格' if strength_qualified else '不合格'),
                                            ('合格' if overall_stability_qualified else '不合格'), 
                                            ('合格' if qualified else '不合格')), file=results_file)

    if qualified:
        # print(np.around((c0_b2, c0_t1, c0_t2, c0_b1), decimals=3))
        # print(np.around((c1_b2, c1_t1, c1_t2, c1_b1), decimals=3))
        # print(np.around((c2_b2, c2_t1, c2_t2, c2_b1), decimals=3))
        # print(np.around((c3_b2, c3_t1, c3_t2, c3_b1), decimals=3))
        # print(np.around((c4_b2, c4_t1, c4_t2, c4_b1), decimals=3))
        # print(np.around((c5_b2, c5_t1, c5_t2, c5_b1), decimals=3))
        # print(np.around((c6_b2, c6_t1, c6_t2, c6_b1), decimals=3))
        current_case = (
            (c0_b2, c0_t1, c0_t2, c0_b1), 
            (c1_b2, c1_t1, c1_t2, c1_b1), 
            (c2_b2, c2_t1, c2_t2, c2_b1), 
            (c3_b2, c3_t1, c3_t2, c3_b1), 
            (c4_b2, c4_t1, c4_t2, c4_b1), 
            (c5_b2, c5_t1, c5_t2, c5_b1), 
            (c6_b2, c6_t1, c6_t2, c6_b1)
        )
        
        qualified_count += 1
        # print(np.around(current_case, decimals=3), file=results_file)
        qualified_cases.append(current_case)
        
    # else:
    #     print('不合格\n')

                            


print('共 %d 种合格'% (qualified_count))

qualified_cases = np.array(qualified_cases)
np.save('stage2_qualified_cases.npy', qualified_cases)                           
results_file.close()