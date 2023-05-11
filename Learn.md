# 學習Pytorch

這是以偏向數學的角度寫下來的，所以會有比較多的數學描述  
在這篇文章中將會把運算建立在向量空間上，並以向量做計算

---
## 核心概念與基本元素
---
### 神經網路(Neural Network)

在使用Pytorch之前，先來點簡單的神經網路(Neural Network, NN)的概念以及數學操作式  
一個最簡單的神經網路有幾個名詞要去認識：神經元、輸入層、隱藏層、輸出層、損失函數  
這邊假設有$N$個$n$維度的向量 $\vec{x_i} = (x_{i1}, x_{i2}, ..., x_{in}) \in \mathbb{R^n}$ 作為輸入

* 神經元(Neuron)
  * 神經網路的最基本運算單位，作用是對輸入做運算得到結果(廢話)
  * 在神經元裡面會有幾個要知道的名詞：<u>權重(weight)</u>、<u>偏誤(Bias)</u>、<u>活化函數(Activation Function)</u>
    * 權重(Weight)：矩陣，會做矩陣運算
    * 偏誤(Bias)：向量，同矩陣運算後結果的長度
    * 活化函數(Activation Function)：將運算結果轉成機率分布的樣子
  

* 輸入層(Input Layer)
  *  就是一層接收你輸入的地方

---

Pytorch 是以張量(Tensor)當最基本的運算元素，可以想像成是高維的向量便於理解，下面簡單介紹一些基本的語法與解釋


Note: 
```python
   import torch 
```


* torch.tensor()