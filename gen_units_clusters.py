#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict

import numpy as np
from sklearn.cluster import KMeans
import json

from bridge_base import ZkLoad
from bridge import Bridge_160


# ### 输入参数
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
### 载入数据
bridge = Bridge_160()
bridge.load_data(path)
bridge.load_params(E=E, P=P, h=h, 
                   bottom_chord_nodes_nums=bottom_chord_nodes_nums,
                   bottom_chord_length=bottom_chord_length, 
                   load=zk_load)




print('检算 => 初始化截面参数')
units_group = bridge.units_nums
# 梁截面数据([腹板宽度 翼缘厚度 腹板厚度 翼缘宽度])
#             b2       t1      t2     b1

b1, t1, b2, t2 = 0.215, 0.012, 0.436, 0.010
beam_section_data = [b2, t1, t2, b1]
print(beam_section_data)

bridge.check()
bridge.set_section_params(units_group, beam_section_data)
bridge.update()



target_units_nums = list(range(1, 40))
target_units_max_forces = [list(bridge.units[unit_num].max_forces) for unit_num in target_units_nums]


n_clusters = 7
kmeans = KMeans(n_clusters, max_iter=300, random_state=0).fit_predict(np.array(target_units_max_forces))


units_clusters = OrderedDict({label: [] for label in range(n_clusters)})
for (unit_num, label) in zip(target_units_nums, kmeans):
    units_clusters[label].append(unit_num)

'''
clusters == (
    (0, [2, 5, 6, 13, 17, 25, 26, 28, 30, 32]), 
    (1, [38]), 
    (2, [1, 4, 8, 9, 21, 29, 34]), 
    (3, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]), 
    (4, [37]), 
    (5, [10, 14, 18, 22, 33, 36]), 
    (6, [12, 16, 20, 24])
)
'''
units_clusters[2] += [43, 47, 51, 55, 59, 63, 67, 71, 75]
units_clusters[1] += [42]
units_clusters[5] += [41]
units_clusters[6] += [44, 48, 53, 61, 65, 73, 74, 77]
units_clusters[0] += [40, 45, 50, 54, 58, 62, 66, 70]
units_clusters[3] += [52, 56, 60, 64]
units_clusters[4] += [46, 49, 57, 68, 69, 72, 76]

for i, cluster in units_clusters.items():
    print(i, cluster)

with open('units_clusters.json', 'w') as units_clusters_file:                                
    json.dump(units_clusters, units_clusters_file)                             











