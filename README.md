# 學習Pytorch

紀錄一下學習機器學習的內容及過程

---
## 核心概念與基本元素
---
### 神經網路(Neural Network)概念介紹與名詞解釋

在使用Pytorch之前，先來點簡單的神經網路(Neural Network, NN)的概念以及數學操作式  
一個最簡單的神經網路有幾個名詞要去認識：神經元、輸入層、隱藏層、輸出層、損失函數、基準真相  
一個神經網路的運作流向為：
$$
\text{Inputs} \rightarrow 輸入層 \rightarrow 隱藏層們 \rightarrow 輸出層 \rightarrow \text{Outputs} \left( \stackrel{Loss Function}{\Longleftrightarrow} \text{基準真相} \right)
$$
這邊假設有 $N$ 個 $n$ 維度的向量 $\vec{x_i} = (x_{i1}, x_{i2}, ..., x_{in}) \in \mathbb{R^n}$ 作為輸入

* 神經元(Neuron)
  * 神經網路的最基本運算單位，作用是對輸入做運算得到結果(廢話)
  * 在神經元裡面會有幾個要知道的名詞：<u>權重(Weight)</u>、<u>偏誤(Bias)</u>、<u>活化函數(Activation Function)</u>
    * 權重(Weight, W)：矩陣，會做矩陣運算
    * 偏誤(Bias, b)：向量，同矩陣運算後結果的長度
    * 活化函數(Activation Function)：將運算結果轉成機率分布的樣子，對向量的每個分量去做活化，可以依據需求選擇 ReLU, softmax, sigmoid...
    * 計算式如下
    $$
    \text{activation}(Wx+b)
    $$

  
* 輸入層(Input Layer)
  * 就是一層接收你輸入的地方
  * 這一層 __沒有計算__，單純就是接收訊息然後傳遞進網路內做運算

  
* 隱藏層(Hidden Layer)
  * 人們所說的黑盒子的部分
  * 由神經元們組成，主要做計算的地方，算法參考神經元的部分
  * 接收前一層(可以是輸入層也可以是其他的隱藏層)的輸出並做計算
  * 與前一層的連接可以全連接(Fully connected, 每個神經元前後都有連好連滿)，也可以依照需求做部分神經元的連接
  * 隱藏層需要設定活化函數，可以依照需求讓每層的活化函數不一樣
  * 隱藏層會根據每一層的神經元數量來做到維度的變化
  * 所以假設今天有一層50跟一層100神經元的隱藏層，維度會先從50變成100，而且在計算的時候就是做150次的 矩陣相乘+活化


* 輸出層(Output Layer)
  * 整個神經網路的最後一層，也是由神經元們組成
  * 會做最後一次的計算後給出最後的輸出
  * 依照需求做出對應數量的輸出
  * 舉例來說，如果是做N類分類問題的話輸出就是一個長度為N的向量，每個分量代表屬於該分類的機率有多大

* 基準真相(Ground Truth)
  * 預期會出現的結果，也就是理論的輸出
  * 在最一開始的時候就會先訂好了
  * 會跟模型輸出做比較進而對模型做訓練以及評估模型表現

* 損失函數(Loss Function)
  * 用來計算每個輸出與預期結果之間的差距
  * 可選用的函數也很多，依據需求作選用
  * 像是在分類問題會選用交叉熵(Cross Entropy)，或是當模型會輸出向量時則會選用 $l_2$ norm
  * 也可以依據自己的需求做客製函數
  * 當損失函數的值越大代表模型輸出與期望目標的差距越大
  * 給模型一個表現狀況的評估參考

---

### Pytorch 的程式函數

Pytorch 是以張量(Tensor)當最基本的運算元素，可以想像成是高維的向量便於理解，下面簡單介紹一些基本的語法與解釋


Note: 
```python
   import torch 
```


* torch.tensor(*input, dtype=, device=)
  * 宣告一個Tensor的變數，不論是直接宣告或是講原有參數做轉換
  * input 可以放 list, np.array等，就是把樹入轉成Tensor
  * dtype: 指定變數的資料型別，有時會因型別不同的資料沒辦法做計算，要注意
  * device: 指頂目前這個變數要放在哪裡， CPU或GPU

* torch.nn
* torch.nn.Module
* torch.nn.Linear

```python
import torch
import troch.nn as nn
```