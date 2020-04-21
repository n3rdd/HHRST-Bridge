#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict

import numpy as np
import json

from bridge_base import ZkLoad
from bridge import Bridge_160

########################### 载入数据 ################################
# 常量
E = 21 * 10**10           # 杨氏模量 (Pa)
P = 1                     # 外力    (kN)
h = 2.52                  # 恒载    (kN)

bottom_chord_length = 8.   # 下弦杆长度/下节点间距 (m)

bridge_len = 160

# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 40]
bottom_chord_nodes_nums = list(range(1, 40, 2)) + [40]

zk_load = ZkLoad(bridge_len)
path = './data/'

bridge = Bridge_160()
bridge.load_data(path)
bridge.load_params(E=E, P=P, h=h, 
                   bottom_chord_nodes_nums=bottom_chord_nodes_nums,
                   bottom_chord_length=bottom_chord_length, 
                   load=zk_load)


with open('units_clusters.json', 'r') as units_clusters_file:
    units_clusters = json.load(units_clusters_file)




########################### 第一阶段检算 ################################
bridge.check()
qualified_count = 0
qualified_cases = []
count = 0
results_file = open('stage_1_results.txt', 'w')

# b = 2 * t1 + b2 = 0.46
# h = 2 * b1 + t2 = 0.76
c0_b = c1_b = c2_b = c3_b = c4_b = c5_b = c6_b = 0.46

c0_h = 0.60
c1_h = 0.76
c2_h = 0.44
c3_h = 0.44
c4_h = 0.76
c5_h = 0.60
c6_h = 0.60

c0_t2 = c1_t2 = c2_t2 = c3_t2 = c4_t2 = c5_t2 = c6_t2 = 0.022

t1_range = np.arange(0.03, 0.03 + 0.01, 0.01)

for c0_t1 in t1_range:
    c0_b2 = c0_b - 2 * c0_t1
    c0_b1 = (c0_h - c0_t2) / 2

    for c1_t1 in t1_range:
        c1_b2 = c1_b - 2 * c1_t1
        c1_b1 = (c1_h - c1_t2) / 2

        for c2_t1 in t1_range:
            c2_b2 = c2_b - 2 * c2_t1
            c2_b1 = (c2_h - c2_t2) / 2

            for c3_t1 in t1_range:
                c3_b2 = c3_b - 2 * c3_t1
                c3_b1 = (c3_h - c3_t2) / 2

                for c4_t1 in t1_range:
                    c4_b2 = c4_b - 2 * c4_t1
                    c4_b1 = (c4_h - c4_t2) / 2

                    for c5_t1 in t1_range:
                        c5_b2 = c5_b - 2 * c5_t1
                        c5_b1 = (c5_h - c5_t2) / 2

                        for c6_t1 in t1_range:
                            c6_b2 = c6_b - 2 * c6_t1
                            c6_b1 = (c6_h - c6_t2) / 2

                            
                            beam_section_0 = [c0_b2, c0_t1, c0_t2, c0_b1]
                            beam_section_1 = [c1_b2, c1_t1, c1_t2, c1_b1]
                            beam_section_2 = [c2_b2, c2_t1, c2_t2, c2_b1]
                            beam_section_3 = [c3_b2, c3_t1, c3_t2, c3_b1]
                            beam_section_4 = [c4_b2, c4_t1, c4_t2, c4_b1]
                            beam_section_5 = [c5_b2, c5_t1, c5_t2, c5_b1]
                            beam_section_6 = [c6_b2, c6_t1, c6_t2, c6_b1]
                            bridge.set_section_params(units_clusters['0'], beam_section_0)
                            bridge.set_section_params(units_clusters['1'], beam_section_1)
                            bridge.set_section_params(units_clusters['2'], beam_section_2)
                            bridge.set_section_params(units_clusters['3'], beam_section_3)
                            bridge.set_section_params(units_clusters['4'], beam_section_4)
                            bridge.set_section_params(units_clusters['5'], beam_section_5)
                            bridge.set_section_params(units_clusters['6'], beam_section_6)           
                            
                            bridge.update()

                            
                            stiffness_qualified = bridge.stiffness_check()
                            local_stability_qualified = bridge.local_stability_check()
                            fatigue_qualified = bridge.fatigue_check()
                            
                            qualified = np.array((stiffness_qualified, 
                                                 local_stability_qualified, 
                                                 fatigue_qualified)).all()

                            
                            count += 1
                            print('%2d  (刚度)%s  (局部稳定)%s  (疲劳)%s  %s\n' % 
                            				(  count,  
                                              ('合格' if stiffness_qualified else '不合格'),
                                              ('合格' if local_stability_qualified else '不合格'),
                                              ('合格' if fatigue_stability_qualified else '不合格'), 
                                              ('合格' if qualified else '不合格')))

                            print('%2d  (刚度)%s  (局部稳定)%s  (疲劳)%s  %s\n' % 
                            				(  count,  
                                              ('合格' if stiffness_qualified else '不合格'),
                                              ('合格' if local_stability_qualified else '不合格'),
                                              ('合格' if fatigue_stability_qualified else '不合格'), 
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
                                qualified_cases.append(current_case)

                                

                           
print('共 %d 种合格'% (qualified_count))

qualified_cases = np.array(qualified_cases)
np.save('stage_1_qualified_cases.npy', qualified_cases)                           
results_file.close()
