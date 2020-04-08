# HHRST-Bridge 
  
## 简介
重载铁路连续钢桁梁桥 (Heavy Haul Railway Steel Truss Bridge) 的初步设计项目，以我国铁路钢桥的标准设计结构为背景，完成结构设计、结构计算与检算的全过程。
### 1. 结构设计
64米 / 160米
### 2. 结构计算
计算获得桥梁结构在各种荷载组合下的受力及变形情况，得出杆件的内力为下一步设计工作提供理论依据。
### 3. 结构检算
根据计算得到的内力对杆件疲劳、刚度、稳定进行检算，从而判断设计尺寸是否满足要求。  



## 安装
下载[python](https://www.python.org/)后在终端中安装 numpy 和 jupyter notebook 包。  
```sh 
$ pip install numpy jupyter notebook
```
  


## 演示
在终端中打开[demo.ipynb](demo-v5.1.ipynb)，运行可获取计算过程及结果。
```sh 
$ jupyter notebook demo.ipynb
``` 

1. 计算总体刚度矩阵
2. 计算节点竖向位移 => 节点竖向位移影响线
3. 计算杆件单元轴力 => 杆件单元轴力影响线
4. zk活载
5. 检验
  


## 贡献者
Chen Lecong @ HDU (Code), Jiang Linhuang @ CSU (Design)



## 使用许可

[MIT](LICENSE) © Chen Lecong
