# 禽流感病毒(H5N1)预测模型项目文档

## 一、项目概述

本项目旨在构建一个基于机器学习的禽流感病毒(H5N1)预测模型，利用历史数据对禽流感的发生情况进行预测。通过对社交媒体上疫情相关信息的采集和分析，结合机器学习算法，实现对禽流感疫情的有效监测和预测。本项目由上饶满星科技有限公司 AI 研究团队开发，包含数据采集、模型训练、预测和可视化等多个环节。

## 二、项目结构

### 2.1 文件夹结构

```
avian_influenza_classification/
|-- data/
|   |-- avian_influenza_(HPAI)_dataset.csv  # 训练数据集
|-- output/
|   |-- 模型评估报告_*.txt  # 模型评估报告
|   |-- 模型评估指标_*.png  # 评估指标可视化图表
|   |-- 预测对比折线图_*.png  # 实际值与预测值对比折线图
|   |-- 预测结果_*.csv  # 预测结果
|   |-- social_media_data/
|       |-- 疫情数据_*.xlsx  # 社交媒体采集的疫情数据（Excel格式）
|       |-- 疫情数据_*.json  # 社交媒体采集的疫情数据（JSON格式）
|   |-- plots/
|       |-- 混淆矩阵_*.png  # 混淆矩阵图
|       |-- 类别分布对比_*.png  # 类别分布对比图
|       |-- 特征重要性_*.png  # 特征重要性图
|-- src/
|   |-- __init__.py
|   |-- classifier.py  # 禽流感分类器类实现
|-- get_news.py  # 社交媒体疫情数据采集脚本
|-- main.py  # 主程序入口
|-- config.json  # 配置文件
```

### 2.2 文件功能说明

- **data/avian*influenza*(HPAI)\_dataset.csv**：包含用于训练模型的历史禽流感数据，是模型训练的基础。
- **get_news.py**：负责从社交媒体（主要是 Twitter）上采集与禽流感疫情相关的信息，并将采集到的数据保存为 Excel 和 JSON 格式。
- **src/classifier.py**：定义了`AvianInfluenzaClassifier`类，该类实现了数据预处理、模型训练、预测和可视化等功能。采用决策树算法和 SMOTE 过采样技术处理不平衡数据。
- **main.py**：项目的主程序入口，负责调用`AvianInfluenzaClassifier`类进行模型训练、预测，并生成评估报告和可视化图表。
- **config.json**：配置文件，可用于设置数据路径、采集推文数量、日志级别等参数。

## 三、技术原理

### 3.1 数据采集

`get_news.py`脚本使用`snscrape`库从 Twitter 上采集与指定关键词相关的推文。通过构造查询语句，结合关键词和语言限制，获取符合条件的推文信息，包括推文内容、发布时间、点赞数、转发数等。采集到的数据会保存到`output/social_media_data`目录下的 Excel 和 JSON 文件中。

### 3.2 数据预处理

在`src/classifier.py`中，`preprocess_data`方法对输入的数据进行预处理。对于数据集中的分类变量，使用`LabelEncoder`进行编码，将其转换为数值型变量，以便后续的机器学习算法处理。

### 3.3 模型训练

`AvianInfluenzaClassifier`类中的`train_model`方法使用决策树算法进行模型训练。首先，将数据集划分为训练集和测试集，采用`train_test_split`函数，测试集占比为 20%。由于数据可能存在类别不平衡问题，使用`SMOTE`（Synthetic Minority Over - sampling Technique）过采样技术对少数类进行合成样本，以平衡数据集。然后，使用决策树分类器对处理后的训练集进行训练。

决策树是一种基于树结构进行决策的机器学习算法，它通过对特征空间进行递归划分，构建决策规则。对于每个内部节点，根据某个特征的取值进行划分，直到达到叶节点，叶节点表示最终的分类结果。决策树的数学原理基于信息论，常用的划分准则有信息增益、信息增益比和基尼指数等。在本项目中，使用默认的划分准则进行决策树的构建。

### 3.4 模型评估

在训练完成后，使用测试集对模型进行评估。计算预测准确率、精确率、召回率和 F1 - 分数等评估指标，并生成分类报告。这些指标的计算公式如下：

- **准确率（Accuracy）**：$Accuracy=\frac{TP + TN}{TP+TN+FP+FN}$
  其中，$TP$（True Positives）表示真正例，即实际为正类且被预测为正类的样本数；$TN$（True Negatives）表示真反例，即实际为反类且被预测为反类的样本数；$FP$（False Positives）表示假正例，即实际为反类但被预测为正类的样本数；$FN$（False Negatives）表示假反例，即实际为正类但被预测为反类的样本数。
- **精确率（Precision）**：$Precision=\frac{TP}{TP + FP}$
  精确率衡量了模型预测为正类的样本中，实际为正类的比例。
- **召回率（Recall）**：$Recall=\frac{TP}{TP + FN}$
  召回率衡量了模型能够正确识别出的正类样本的比例。
- **F1 - 分数（F1 - Score）**：$F1=\frac{2\times Precision\times Recall}{Precision + Recall}$
  F1 - 分数是精确率和召回率的调和平均值，综合考虑了两者的性能。

### 3.5 预测与可视化

使用训练好的模型对新数据进行预测，通过`make_predictions`方法实现。同时，项目提供了多种可视化功能，包括绘制混淆矩阵图、类别分布对比图、特征重要性图以及评估指标折线图和实际值与预测值对比折线图等。这些可视化图表有助于直观地理解模型的性能和预测结果。

## 四、使用方法

### 4.1 环境配置

确保已经安装了以下 Python 库：

- `pandas`：用于数据处理和分析。
- `numpy`：用于数值计算。
- `scikit - learn`：包含机器学习算法和评估指标。
- `imblearn`：用于处理类别不平衡问题。
- `snscrape`：用于社交媒体数据采集。
- `matplotlib`：用于数据可视化。
- `seaborn`：用于美化可视化图表。

可以使用以下命令安装所需库：

```bash
pip install pandas numpy scikit-learn imblearn snscrape matplotlib seaborn
```

### 4.2 配置文件设置

在`config.json`文件中设置数据路径、采集推文数量、日志级别等参数。例如：

```json
{
  "training_data_path": "./data/avian_influenza_(HPAI)_dataset.csv",
  "new_data_path": "output/new_data.csv",
  "keywords": ["禽流感", "H5N1", "疫情", "outbreak"],
  "num_tweets": 1000,
  "lang": "zh",
  "logging_level": "INFO"
}
```

### 4.3 数据采集

运行`get_news.py`脚本进行社交媒体数据采集：

```bash
python get_news.py
```

采集到的数据将保存到`output/social_media_data`目录下。

### 4.4 模型训练与预测

运行`main.py`脚本进行模型训练、预测和可视化：

```bash
python main.py
```

脚本将自动完成数据加载、模型训练、评估、预测和可视化等步骤，并将结果保存到`output`目录下。

## 五、注意事项

- **数据文件路径**：确保`config.json`中设置的数据文件路径正确，否则可能会导致文件读取错误。
- **类别不平衡问题**：虽然使用了`SMOTE`过采样技术处理类别不平衡问题，但在某些情况下，可能仍然需要进一步调整模型或采样策略。
- **日志记录**：项目使用日志记录程序的执行过程和错误信息，日志文件名为`avian_influenza.log`，可根据需要查看日志文件进行调试和监控。
- **中文字体显示**：如果在可视化图表中遇到中文字体显示问题，可根据操作系统不同，修改`main.py`中的字体配置，如在 Windows 系统中使用`SimHei`字体，在 macOS 系统中使用`WenQuanYi Zen Hei`字体。

## 六、总结

本项目通过数据采集、预处理、模型训练、评估和可视化等步骤，实现了一个基于机器学习的禽流感病毒(H5N1)预测模型。利用社交媒体数据和决策树算法，能够对禽流感疫情进行有效的监测和预测。通过可视化图表，可以直观地了解模型的性能和预测结果，为疫情防控提供有力的支持。同时，项目具有良好的可扩展性，可以通过调整配置文件和模型参数，进一步优化模型性能。


### 七、数学与科学算法深入解析

#### 7.1 数据预处理中的编码理论

在数据预处理阶段，使用`LabelEncoder`对分类变量进行编码。从数学角度来看，这是一种将离散的分类值映射到连续整数的操作。假设我们有一个分类变量 $C$，其取值集合为 $\{c_1, c_2, \cdots, c_n\}$，`LabelEncoder`会为每个取值 $c_i$ 分配一个唯一的整数 $i$（$i = 0, 1, \cdots, n - 1$）。这种映射关系可以表示为一个函数 $f: C \to \mathbb{Z}$，其中 $\mathbb{Z}$ 是整数集。

在信息论中，这种编码方式有助于减少数据的冗余，使得数据能够以更紧凑的形式存储和处理。例如，对于一个具有 $n$ 个不同取值的分类变量，使用 $k$ 位二进制编码可以表示 $2^k$ 个不同的状态。当 $2^{k - 1}<n\leqslant2^k$ 时，我们可以使用 $k$ 位二进制数来编码这 $n$ 个取值，从而实现数据的高效表示。

#### 7.2 决策树算法的数学基础

决策树是一种基于树结构进行决策的机器学习算法，其核心在于如何选择最优的划分特征和划分点。常用的划分准则有信息增益、信息增益比和基尼指数。

##### 7.2.1 信息增益

信息增益是基于信息熵的概念。信息熵是对信息不确定性的度量，对于一个离散随机变量 $X$，其取值集合为 $\{x_1, x_2, \cdots, x_n\}$，概率分布为 $P(X = x_i)=p_i$（$i = 1, 2, \cdots, n$），则信息熵 $H(X)$ 的定义为：

$$
\[H(X)=-\sum\_{i = 1}^{n}p_i\log_2p_i\]
$$

在决策树中，我们希望通过选择某个特征进行划分，使得划分后的子集的信息熵尽可能降低。假设数据集 $D$ 有 $m$ 个样本，属于 $k$ 个不同的类别，第 $j$ 个类别的样本数为 $m_j$，则数据集 $D$ 的信息熵为：
\[H(D)=-\sum\_{j = 1}^{k}\frac{m_j}{m}\log_2\frac{m_j}{m}\]

对于特征 $A$，其取值集合为 $\{a_1, a_2, \cdots, a_v\}$，根据特征 $A$ 的取值将数据集 $D$ 划分为 $v$ 个子集 $D_1, D_2, \cdots, D_v$，子集 $D_i$ 中的样本数为 $m_i$。则在特征 $A$ 条件下，数据集 $D$ 的条件熵 $H(D|A)$ 为：
\[H(D|A)=\sum\_{i = 1}^{v}\frac{m_i}{m}H(D_i)\]

信息增益 $g(D, A)$ 定义为数据集 $D$ 的信息熵与在特征 $A$ 条件下的条件熵之差：
$$
\[g(D, A)=H(D)-H(D|A)\]
$$

决策树在选择划分特征时，会选择信息增益最大的特征作为当前节点的划分特征。

##### 7.2.2 信息增益比

信息增益存在一个问题，即它倾向于选择取值较多的特征。为了解决这个问题，引入了信息增益比。特征 $A$ 对数据集 $D$ 的信息增益比 $g_R(D, A)$ 定义为信息增益 $g(D, A)$ 与特征 $A$ 的固有值 $IV(A)$ 之比：
$$
\[IV(A)=-\sum\_{i = 1}^{v}\frac{m_i}{m}\log_2\frac{m_i}{m}\]
\[g_R(D, A)=\frac{g(D, A)}{IV(A)}\]
$$
##### 7.2.3 基尼指数

基尼指数是另一种衡量数据不纯度的指标。对于数据集 $D$，其基尼指数 $Gini(D)$ 定义为：
\[Gini(D)=1-\sum\_{j = 1}^{k}(\frac{m_j}{m})^2\]

对于特征 $A$ 的某个取值 $a$，将数据集 $D$ 划分为 $D_1$ 和 $D_2$ 两个子集，则在特征 $A$ 取值为 $a$ 时的

## 许可证

本项目采用 **木兰宽松许可证 (Mulan PSL)** 进行许可。  
有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

[![License: Mulan PSL v2](https://img.shields.io/badge/License-Mulan%20PSL%202-blue.svg)](http://license.coscl.org.cn/MulanPSL2)

### 个人捐赠支持

如果您认为该项目对您有所帮助，并且愿意个人捐赠以支持其持续发展和维护，🥰 我非常感激您的慷慨。
您的捐赠将帮助我继续改进和添加新功能到该项目中。 通过 financial donation，您将有助于确保该项目保持免
费和对所有人开放。即使是一小笔捐款也能产生巨大的影响，也是对我个人的鼓励。

### 国内支付方式

<div align="center">
<table>
<tr>
<td align="center" width="300">
<img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9863.jpg?raw=true" width="200" />
<br />
<strong>微信支付</strong>
</td>
<td align="center" width="300">
<img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9859.JPG?raw=true" width="200" />
<br />
<strong>支付宝</strong>
</td>
</tr>
</table>
</div>

### 国际支付渠道

<div align="center">

[![支付宝](https://img.shields.io/badge/支付宝-捐赠-00A1E9?style=for-the-badge&logo=alipay&logoColor=white)](https://qr.alipay.com/fkx19369scgxdrkv8mxso92)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-赞助-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/F1F5VCZJU)
[![PayPal](https://img.shields.io/badge/PayPal-支持-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://www.paypal.com/paypalme/ctkqiang)
[![Stripe](https://img.shields.io/badge/Stripe-捐赠-626CD9?style=for-the-badge&logo=Stripe&logoColor=white)](https://donate.stripe.com/00gg2nefu6TK1LqeUY)

</div>

### 关注作者

<div align="center">

#### 专业平台

[![GitHub](https://img.shields.io/badge/GitHub-开源项目-24292e?style=for-the-badge&logo=github)](https://github.com/ctkqiang)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-职业经历-0077b5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ctkqiang/)
[![Stack Overflow](https://img.shields.io/badge/Stack_Overflow-技术交流-f48024?style=for-the-badge&logo=stackoverflow)](https://stackoverflow.com/users/10758321/%e9%92%9f%e6%99%ba%e5%bc%ba)

#### 社交媒体

[![Facebook](https://img.shields.io/badge/Facebook-社交平台-1877F2?style=for-the-badge&logo=facebook)](https://www.facebook.com/JohnMelodyme/)
[![Instagram](https://img.shields.io/badge/Instagram-生活分享-E4405F?style=for-the-badge&logo=instagram)](https://www.instagram.com/ctkqiang)
[![Twitch](https://img.shields.io/badge/Twitch-直播频道-9146FF?style=for-the-badge&logo=twitch)](https://twitch.tv/ctkqiang)

[![](https://img.shields.io/badge/GitHub-项目仓库-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ctkqiang)
[![](https://img.shields.io/badge/微信公众号-华佗AI-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9245.JPG?raw=true)

</div>
