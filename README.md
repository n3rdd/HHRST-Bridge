# HHRST-Bridge 

## 简介
以我国铁路钢桥的标准设计结构为背景，实现重载铁路连续钢桁梁桥 (Heavy Haul Railway Steel Truss Bridge) 的初步设计，项目包括结构设计、结构计算与检算的全过程。
#### 1. 结构设计
<div align=center>
<img src=64m_small.jpg></img>
<p><b>图1. 64米结构</b></p>
</div>


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
在终端中打开 [Demo.ipynb](Demo-v5.3.ipynb) ，运行可获取计算过程及结果。
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

- zk活载对称滑动是否需要修改
- 每个时刻节点位移/杆件轴力添加是否对应




## 贡献者

Chen Lecong @ HDU (Code), Jiang Linhuang @ CSU (Design)



## 使用许可

[MIT](LICENSE)
