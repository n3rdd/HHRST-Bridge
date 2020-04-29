# SBDH: Steel Bridge Design Helper
让初级钢桥设计者的生活更简单！

## 简介
项目用Python构建，提供钢桥设计过程中计算与检算所需的工具，支持___结构。

<div align=center>
<img src=https://user-images.githubusercontent.com/26713883/79548517-29200400-80c8-11ea-99a6-79b7bb52c2f8.jpg></img>
<p><b>图1. 64米结构</b></p>
\#
<p><b>图1. 160米结构</b></p>
</div>

项目适用于土木工程专业本科高年级或研究生低年级的同学作为辅助工具使用。

#### 2. 结构计算  
计算获得桥梁结构在各种荷载组合下的受力及变形情况，得出杆件的内力，为下一步设计工作提供理论依据。
#### 3. 结构检算 
根据计算得到的内力对杆件疲劳、刚度、稳定进行检算，从而判断设计尺寸是否满足要求。  



## 安装
下载 [Python](https://www.python.org/) 后在终端中安装 [Numpy](https://www.numpy.org.cn/) 和 [Jupyter Notebook](https://jupyter.org/) 包。  
```sh 
$ pip install numpy notebook
```



## 演示
在终端中打开 [Demo.ipynb](Demo-v2.2.3.ipynb) ，运行可获取计算过程及结果。
```sh 
$ jupyter notebook demo.ipynb
```

1. 计算总体刚度矩阵
2. 计算节点竖向位移 => 节点竖向位移影响线
3. 计算杆件单元轴力 => 杆件单元轴力影响线
4. zk活载
5. 检验
  



## API

todo



## 更新日志

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

#### TODO:

- get_units_axial_forces_moment需检查和self.units顺序是否对应
- get_one_unit_axial_force_moment中get_u_and_v中node_num应该与units.keys()比较而不是bc_nodes_nums
- setter => 自动计算
- reduce_K 修改
- 添加Checker 类




## 贡献者

Chen Lecong @ HDU (实现), Jiang Linhuang @ CSU (理论支持)



## 使用许可

[MIT](LICENSE)
