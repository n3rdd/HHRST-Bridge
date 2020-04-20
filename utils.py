import numpy as np

### 工具函数 ###
def partition_kij(kij):
    '''单元刚度矩阵kij分块'''
    shape = (2, 2, 2, 2)

    # 大行之间隔 8 个元素，大列之间隔 2 个元素
    # 小行之间隔 4 个元素，小列之间隔 1 个元素
    strides = kij.itemsize * np.array([8, 2, 4, 1])
    squares = np.lib.stride_tricks.as_strided(kij, shape=shape, strides=strides)
    return squares


def seat_kij(K, unit, kij):
    '''更新整体刚度矩阵
    参数：整体刚度矩阵K, 杆件单元, 单元刚度矩阵
    返回：更新后整体刚度矩阵
    '''
    p_kij = partition_kij(kij)  # patitioned_kij 4*4矩阵 => 分块成 2*2*2*2
    i, j = unit.nodes_pair[0].num, unit.nodes_pair[1].num
    i, j = i - 1, j - 1
    K[i][i] += p_kij[0][0]
    K[i][j] += p_kij[0][1]
    K[j][i] += p_kij[1][0]
    K[j][j] += p_kij[1][1]

    return K


def reshape_K(K, n):
    '''
    改变总体刚度矩阵形状
    输入：总体刚度矩阵K, n个节点
    如：n = 16: 16*16*2*2矩阵 => 32*32矩阵
    '''
    v_l = []  # 存2*18矩阵
    for i in range(n):
        h_l = [K[i][j] for j in range(n)]
        v_l.append(np.hstack(h_l))
    return np.vstack(v_l)


def reduce_K(K, bridge_len):
    '''计算位移和轴力
    '''
    assert bridge_len in [64, 160], 'Bridge length not in [64, 160].'
    
    # 去掉1行1列，2行2列，32行32列
    if bridge_len == 64:
        #return K[2:31, 2:31]
        return np.delete(np.delete(K, [0, 1, 31], axis=0), [0, 1, 31], axis=1)
    
    # 去掉1行1列，2行2列，42行42列，80行80列
    elif bridge_len == 160:
        return np.delete(np.delete(K, [0, 1, 41, 79], axis=0), [0, 1, 41, 79], axis=1)

def pretty_print(G, nrow, ncol):
    '''矩阵美观打印'''
    for i in range(nrow):
        for j in range(ncol):
            if(j % 2 == 0):
                print('|%-6.2f' % G[i][j], end = '')
            else:
                print('%-6.2f' % G[i][j], end = '')
        
        if((i + 1) % 2 == 0):
            print('\n')
        else:
            print('')