# PyTorch的引入



## 官方文档网址

学习指南：
https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
API文档：
https://docs.pytorch.org/docs/stable/pytorch-api.html



## PyTorch的典型应用领域

- **计算机视觉（CV）**：图像分类、目标检测、生成模型（如 Stable Diffusion）。
- **自然语言处理（NLP）**：Transformer、LLM（如 GPT、BERT 等）。
- **强化学习（RL）**：与 Gym、RLlib 等结合实现智能体训练。
- **科学计算与量子机器学习**：通过 TorchQuantum 等库扩展。




## PyTorch的用户级API

PyTorch 是一个基于 Python 的科学计算库，它使用动态计算图（即“定义即运行”）机制，为深度学习研究和应用提供了灵活高效的实现。PyTorch 的用户级 API 主要分布在几个核心子模块中，包括 `torch`、`torch.nn`、`torch.optim`、`torch.utils.data` 等。下面分别介绍每个模块存放的主要功能。

1. `torch` 模块

`torch` 是 PyTorch 的核心库，提供了张量（Tensor）的基本定义和操作，类似于 NumPy，但支持 GPU 加速和自动微分。主要功能包括：

- **张量创建和操作**：如 `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()` 等；支持索引、切片、数学运算、线性代数操作等。
- **自动微分**：通过 `torch.autograd` 模块，张量设置 `requires_grad=True` 后，PyTorch 会自动追踪所有操作，并调用 `.backward()` 计算梯度。
- **设备管理**：`torch.device` 用于指定计算设备（CPU/GPU），`tensor.to(device)` 可在设备间迁移数据。
- **随机数生成**：如 `torch.manual_seed()`, `torch.rand()`, `torch.randint()` 等。
- **数学函数**：包括逐元素运算、归约操作（如 `torch.sum()`, `torch.mean()`）、比较操作等。
- **分布式通信**：`torch.distributed` 提供多 GPU 分布式训练的支持。

2. `torch.nn` 模块

`torch.nn` 提供了构建神经网络所需的各种组件，以类和模块化的方式组织，便于定义和复用网络结构。主要功能包括：

- **网络层（Layers）**：如线性层（`nn.Linear`）、卷积层（`nn.Conv2d`）、循环层（`nn.LSTM`）、归一化层（`nn.BatchNorm2d`）、Dropout 等。
- **激活函数**：如 `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.LeakyReLU` 等。
- **损失函数（Loss Functions）**：如 `nn.MSELoss`（均方误差）、`nn.CrossEntropyLoss`（交叉熵损失）、`nn.L1Loss` 等。
- **容器（Containers）**：`nn.Sequential` 用于顺序组合多个层，`nn.ModuleList` 和 `nn.ModuleDict` 用于注册子模块列表或字典。
- **参数管理**：`nn.Parameter` 将张量包装为可注册的参数，自动加入模型的参数列表。
- **实用工具**：如 `nn.utils.clip_grad_norm_` 用于梯度裁剪，`nn.Flatten` 用于展平张量等。

此外，`torch.nn.functional`（通常导入为 `F`）提供了与 `nn` 中类似的函数式接口（如 `F.relu`, `F.conv2d`），但没有内部状态，适用于无参数的操作。

3. `torch.optim` 模块

`torch.optim` 实现了各种优化算法，用于根据计算出的梯度更新模型参数。主要功能包括：

- **优化器类**：如 `optim.SGD`, `optim.Adam`, `optim.RMSprop`, `optim.Adagrad` 等。每个优化器接受一个需要优化的参数列表（通常来自 `model.parameters()`）和特定于算法的超参数（如学习率）。
- **学习率调度**：`torch.optim.lr_scheduler` 子模块提供学习率调整策略，如 `StepLR`, `ReduceLROnPlateau`, `CosineAnnealingLR` 等，可在训练过程中动态调整学习率。
- **参数分组**：优化器支持对不同参数组设置不同的超参数（例如不同层使用不同学习率）。

4. `torch.utils.data` 模块

`torch.utils.data` 提供了数据加载和预处理的工具，支持高效的数据流水线。主要功能包括：

- **Dataset 抽象类**：用户需要继承 `Dataset` 并实现 `__len__` 和 `__getitem__` 方法，以定义自己的数据集。PyTorch 也提供了一些内置数据集（如 `torchvision.datasets` 中的图片数据集）。
- **DataLoader 类**：将 `Dataset` 包装为可迭代对象，支持自动批处理（`batch_size`）、数据打乱（`shuffle`）、多进程加载（`num_workers`）以及自定义采样策略（`sampler`）。
- **Sampler**：定义数据集中索引的采样方式，如顺序采样（`SequentialSampler`）、随机采样（`RandomSampler`）和分布式采样（`DistributedSampler`）。
- **数据转换**：通常与 `torchvision.transforms` 配合使用，对数据进行预处理（如归一化、数据增强）。虽然 `transforms` 属于 `torchvision`，但 `torch.utils.data` 的设计允许将任意转换函数集成到数据加载流程中。

5. 其他相关模块

- **`torchvision`**：虽然不是核心库，但它是 PyTorch 官方提供的计算机视觉工具包，包含常用数据集、模型架构（如 ResNet）、图像转换操作等。
- **`torchtext`**：用于自然语言处理的数据工具包。
- **`torchaudio`**：用于音频处理的工具包。

**总结**

这些模块共同构成了 PyTorch 的用户级 API，形成了一个从数据处理、模型构建、训练优化到推理的完整流程。简单来说：
- `torch` 提供底层张量运算和自动微分；
- `torch.nn` 提供高层神经网络组件；
- `torch.optim` 实现参数更新算法；
- `torch.utils.data` 管理数据加载和预处理。

开发者可以通过组合这些模块，灵活地设计和训练深度学习模型。





## 环境配置

我们使用Anaconda作为Python的包管理工具



**1.Anaconda的安装**

https://www.anaconda.com/download/success

![77183338601](img/1771833386014.png)



安装步骤：一直点击下一步，勾选环节全部选上，放在有空余的盘符上



**2.使用Anaconda创建环境**

创建"pytorchlearn"环境

```
# 打开"Anaconda Prompt"

# 创建环境
create -n pytorchlearn python=3.12

# 查看所有环境，确保环境创建成功
conda env list

# 进入(激活)环境
conda activate pytorchlearn

# 查看当前环境的库与包
conda list

# 退出环境，返回base环境
conda deactivate
```



**3.安装PyTorch**

https://pytorch.org/

在该网址中翻到下面，找到“Previous versions of PyTorch”，不推荐直接下最新版，建议下载较新的版本

之后找到conda支持的下载指令

![77183550076](img/1771835500760.png)



```
# 进入环境
conda activate pytorchlearn

# 下载（最好开启科学上网）
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
```





**4.将环境导入PyCharm**

新建项目，选择Conda，找到你创建的虚拟环境（若没有则重启电脑）

![77183664381](img/1771836643811.png)



测试cpu版本环境

```python
import torch

flag = torch.cuda.is_available()
print(flag) # 返回 true 为cuda安装成功
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)



# # 有GPU的可以另外测试下面的，此处省略
# print(torch.cuda.get_device_name(0))
# print(torch.rand(3, 3).cuda())
# # Check CUDA version
# cuda_version = torch.version.cuda
# print("CUDA Version：", cuda_version)
# # Check CuDNN version
# cuda_version = torch.backends.cudnn.version()
# print("CuDNN Version:", cuda_version)

```

结果：

![77183707462](img/1771837074620.png)





## 实现一个简单的线性神经网络

实现一个简单的线性神经网络，网络函数为$y=2x+1$



分为5步骤：

1.构造数据x，形状为(100,1)即100行1列，对应构造出y=2*x+1+noise，y形状与x的形状相同

2.构建线性模型,nn.Linear(1,1)，输入的特征数为1，输出得标签为1（一个x，一个y）

3.定义损失函数与优化器

4.训练模型: 前向传播=>计算损失=>清空梯度=>反向传播=>更新参数

5.打印出模型的参数w、b(w大致为2，b大致为1)

```python
import torch
from torch import nn, optim

# 1.构造数据
# 1.1.构造特征
x = torch.linspace(-5, 5, 100)  # 生成100个从-5到5,均匀分布的100个点
#print(x, x.shape)  # 构造出100个值，维度是(100)
# tensor([-5.0000, -4.8990, -4.7980, -4.6970, -4.5960, -4.4949, -4.3939, -4.2929,
#         -4.1919, -4.0909, -3.9899, -3.8889, -3.7879, -3.6869, -3.5859, -3.4848,
# 。。。。。。。。。。。。。。。。。。。。。。。。省略
#          3.8889,  3.9899,  4.0909,  4.1919,  4.2929,  4.3939,  4.4949,  4.5960,
#          4.6970,  4.7980,  4.8990,  5.0000])
x = x.unsqueeze(1)  # 把 100列 变成 100行1列，即(100,)->(100,1)
#print(x, x.shape)
# tensor([[-5.0000],
#         [-4.8990],
#         。。。。。。。。。。省略
#         [ 4.7980],
#         [ 4.8990],
#         [ 5.0000]]) torch.Size([100, 1])

# 1.2.生成标签y(y=2*x+1)
y = 2 * x + 1
# 给y添加噪音
noise = torch.randn(y.size())
y += noise
#print(y)
# tensor([[ -8.2897],
#。。。。。。。省略
#         [ 12.7701],
#         [ 12.2594]])

# 2.定义简单的线性模型
model = nn.Linear(in_features=1, out_features=1)

# 3.定义损失函数与优化器
criterion = nn.MSELoss()  # 损失函数：均方误差
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器

# 4.训练模型
epochs = 2000
for epoch in range(1,epochs+1):
    y_pred=model(x)             # 4.1.前向传播
    loss=criterion(y_pred,y)    # 4.2.计算损失
    optimizer.zero_grad()       # 4.3.清空梯度
    loss.backward()             # 4.4.反向传播
    optimizer.step()            # 4.5.更新参数

    print('轮次:{0}，损失值:{1}'.format(epoch,loss))

# 5.查看结果（查看两个参数w、b）
[w,b]=model.parameters(())
print('训练结果:w:{0},b:{1}'.format(w,b))

```

结果截图：

![77192645156](img/1771926451567.png)



## Matplotlib可视化工具

在原来的简单线性神经网络的基础上添加图片可视化

```python
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

# 1.构造数据
# 1.1.构造特征
x = torch.linspace(-5, 5, 100)  # 生成100个从-5到5,均匀分布的100个点
x = x.unsqueeze(1)  # 把 100列 变成 100行1列，即(100,)->(100,1)
# 1.2.生成标签y(y=2*x+1)
y = 2 * x + 1
# 给y添加噪音
noise = torch.randn(y.size())
y += noise

# 2.定义简单的线性模型
model = nn.Linear(in_features=1, out_features=1)

# 3.定义损失函数与优化器
criterion = nn.MSELoss()  # 损失函数：均方误差
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器

# 4.训练模型
epochs = 200
losses = []
weights = []
biases = []
for epoch in range(1, epochs + 1):
    y_pred = model(x)  # 4.1.前向传播
    loss = criterion(y_pred, y)  # 4.2.计算损失
    optimizer.zero_grad()  # 4.3.清空梯度
    loss.backward()  # 4.4反向传播
    optimizer.step()  # 4.5.更新参数

    # 记录损失和参数
    losses.append(loss.item())
    weights.append(model.weight.item())  # 取出标量值
    biases.append(model.bias.item())

    print('轮次:{0}，损失值:{1}'.format(epoch, loss.item()))

# 5.查看结果（查看两个参数w、b）
[w, b] = model.parameters(())
print('训练结果:w:{0},b:{1}'.format(w, b))

# 6.绘制三个关系图
plt.figure(figsize=(12, 4))

# 损失 vs 轮次
plt.subplot(1, 3, 1)
plt.plot(range(1, epochs + 1), losses, color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')

# w vs 轮次
plt.subplot(1, 3, 2)
plt.plot(range(1, epochs + 1), weights, color='red')
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title('Weight vs. Epoch')

# b vs 轮次
plt.subplot(1, 3, 3)
plt.plot(range(1, epochs + 1), biases, color='green')
plt.xlabel('Epoch')
plt.ylabel('Bias (b)')
plt.title('Bias vs. Epoch')

plt.tight_layout()
plt.show()

```

效果图：

![77193949159](img/1771939491598.png)



