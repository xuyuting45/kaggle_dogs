# 实战Kaggle比赛——使用Gluon识别120种狗


我们在本章中选择了Kaggle中的[120种狗类识别问题](https://www.kaggle.com/c/dog-breed-identification)。与之前的[CIFAR-10原始图像分类问题](kaggle-gluon-cifar10.md)不同，本问题中的图片文件大小更接近真实照片大小，且大小不一。本问题的输出也变的更加通用：我们将输出每张图片对应120种狗的分别概率。


## Kaggle中的CIFAR-10原始图像分类问题

[Kaggle](https://www.kaggle.com)是一个著名的供机器学习爱好者交流的平台。为了便于提交结果，请大家注册[Kaggle](https://www.kaggle.com)账号。然后请大家先点击[120种狗类识别问题](https://www.kaggle.com/c/dog-breed-identification)了解有关本次比赛的信息。

![](../img/kaggle-dog.png)



## 整理原始数据集

比赛数据分为训练数据集和测试数据集。训练集包含10,222张图片。测试集包含10,357张图片。

两个数据集都是jpg彩色图片，大小接近真实照片大小，且大小不一。训练集一共有120类狗的图片。



### 下载数据集


登录Kaggle后，数据可以从[120种狗类识别问题](https://www.kaggle.com/c/dog-breed-identification/data)中下载。

* [训练数据集train.zip下载地址](https://www.kaggle.com/c/dog-breed-identification/download/train.zip)

* [测试数据集test.zip下载地址](https://www.kaggle.com/c/dog-breed-identification/download/test.zip)

* [训练数据标签label.csv.zip下载地址](https://www.kaggle.com/c/dog-breed-identification/download/labels.csv.zip)


### 解压数据集

训练数据集train.zip和测试数据集test.zip都是压缩格式，下载后它们的路径可以如下：

* ../data/kaggle_dog/train.zip
* ../data/kaggle_dog/test.zip
* ../data/kaggle_dog/labels.csv.zip

为了使网页编译快一点，我们在git repo里仅仅存放小数据样本（'train_valid_test_tiny.zip'）。执行以下代码会从git repo里解压生成小数据样本。

```{.python .input  n=1}
# 如果训练下载的Kaggle的完整数据集，把demo改为False。
demo = False
data_dir = '../data/kaggle_dog'

if demo:
    zipfiles= ['train_valid_test_tiny.zip']
else:
    zipfiles= ['train.zip', 'test.zip', 'labels.csv.zip']

import zipfile
for fin in zipfiles:
    with zipfile.ZipFile(data_dir + '/' + fin, 'r') as zin:
        zin.extractall(data_dir)
```

### 整理数据集

对于Kaggle的完整数据集，我们需要定义下面的reorg_dog_data函数来整理一下。整理后，同一类狗的图片将出现在在同一个文件夹下，便于`Gluon`稍后读取。

函数中的参数如data_dir、train_dir和test_dir对应上述数据存放路径及原始训练和测试的图片集文件夹名称。参数label_file为训练数据标签的文件名称。参数input_dir是整理后数据集文件夹名称。参数valid_ratio是验证集中每类狗的数量占原始训练集中数量最少一类的狗的数量（66）的比重。

```{.python .input  n=3}
import math
import os
import shutil
from collections import Counter

def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, 
                   valid_ratio):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
    labels = set(idx_label.values())

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    # 训练集中数量最少一类的狗的数量。
    min_num_train_per_label = (
        Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    # 验证集中每类狗的数量。
    num_valid_per_label = math.floor(min_num_train_per_label * valid_ratio)
    label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
```

再次强调，为了使网页编译快一点，我们在这里仅仅使用小数据样本。相应地，我们仅将批量大小设为2。实际训练和测试时应使用Kaggle的完整数据集并调用reorg_dog_data函数整理便于`Gluon`读取的格式。由于数据集较大，批量大小batch_size大小可设为一个较大的整数，例如128。

```{.python .input  n=1}
demo = False
data_dir = '../data/kaggle_dog'
if demo:
    # 注意：此处使用小数据集为便于网页编译。
    input_dir = 'train_valid_test_tiny'
    # 注意：此处相应使用小批量。对Kaggle的完整数据集可设较大的整数，例如128。
    batch_size = 2
else:
    label_file = 'labels.csv'
    train_dir = 'train'
    test_dir = 'test'
    input_dir = 'train_valid_test'
    #batch_size = 128
    #valid_ratio = 0.1 
    #reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir,valid_ratio)
    
```

## 使用Gluon读取整理后的数据集

为避免过拟合，我们在这里使用`image.CreateAugmenter`来加强数据集。例如我们设`rand_mirror=True`即可随机对每张图片做镜面反转。以下我们列举了该函数里的所有参数，这些参数都是可以调的。

```{.python .input  n=2}
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np

def transform_train(data, label):
    im = image.imresize(data.astype('float32') / 255, 300, 300)    
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=300, 
                        rand_crop=True, rand_resize=True, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)   
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))    
    return (im, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 224, 224)
    auglist = [image.ColorNormalizeAug(mean=nd.array([0.485, 0.456, 0.406]), std=nd.array([0.229, 0.224, 0.225]))]
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))
```

接下来，我们可以使用`Gluon`中的`ImageFolderDataset`类来读取整理后的数据集。

```{.python .input  n=3}
input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1, 
                                     transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1, 
                                     transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', # 'Stanford_Dogs_Dataset', 
                                           flag=1, transform=transform_train)
test_stanford_ds = vision.ImageFolderDataset(input_str + 'train_valid', flag=1, 
                                     transform=transform_test)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, 
                                     transform=transform_test)

batch_size = 32
loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, 
                          last_batch='keep')
test_stanford = loader(test_stanford_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 设计模型

我们这里使用了[ResNet-18](resnet-gluon.md)模型。我们使用[hybridizing](../chapter_gluon-advances/hybridize.md)来提升执行效率。

请注意：模型可以重新设计，参数也可以重新调整。

```{.python .input  n=4}
from mxnet.gluon import nn
from mxnet import nd

def Classifier(numout):
    net = nn.HybridSequential()
    net.add(nn.GlobalAvgPool2D())
    net.add(nn.Flatten())
    net.add(nn.Dense(512, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(numout))
    return net

def get_net(ctx):
    from mxnet.gluon.model_zoo import vision as models
    pretrained_net = models.resnet101_v1(pretrained=True)
    featurenet=pretrained_net.features
    #for _, w in featurenet.collect_params().items():
        #w.grad_req = 'null'
    for i in range(6):
        featurenet[i].collect_params().setattr('grad_req', 'null')
    for i in range(6,8):
        featurenet[i].collect_params().setattr('lr_mult', 0.1)    
    
    num_outputs = 120
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(featurenet)
        net.add(Classifier(num_outputs))        
        net[1].collect_params().initialize(init=mx.init.Xavier())
    
    net.collect_params().reset_ctx(ctx)    
    return net
```

## 训练模型并调参

在[过拟合](../chapter_supervised-learning/underfit-overfit.md)中我们讲过，过度依赖训练数据集的误差来推断测试数据集的误差容易导致过拟合。由于图像分类训练时间可能较长，为了方便，我们这里不再使用K折交叉验证，而是依赖验证集的结果来调参。

我们定义损失函数以便于计算验证集上的损失函数值。我们也定义了模型训练函数，其中的优化算法和参数都是可以调的。

```{.python .input  n=5}
import datetime
import sys
sys.path.append('..')
import utils
loss_train=[]
loss_valid=[]
acc_valid=[]

def get_loss(data, net, ctx):
    loss = 0.0
    acc = 0.0
    for feas, label in data:
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        cross_entropy = softmax_cross_entropy(output, label)
        acc += nd.mean(output.argmax(axis=1)==label).asscalar()
        loss += nd.mean(cross_entropy).asscalar()
    return (loss / len(data),acc / len(data))

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, 
          lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 
                                      'wd': wd})
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:  
            valid_loss , valid_acc = get_loss(valid_data, net, ctx)
            loss_valid.append(valid_loss)
            acc_valid.append(valid_acc)
            epoch_str = ("Epoch %d. Train loss: %f, Valid loss %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data), valid_loss, valid_acc))
        else:
            epoch_str = ("Epoch %d. Train loss: %f, "
                         % (epoch, train_loss / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        loss_train.append(train_loss / len(train_data))

```

以下定义训练参数并训练模型。这些参数均可调。为了使网页编译快一点，我们这里将epoch数量有意设为1。事实上，epoch一般可以调大些。

我们将依据验证集的结果不断优化模型设计和调整参数。依据下面的参数设置，优化算法的学习率将在每80个epoch自乘0.1。

```{.python .input  n=6}
import mxnet as mx
ctx = mx.gpu(0)
num_epochs = 60
learning_rate = 0.001
weight_decay = 5e-4
lr_period = 20
lr_decay = 0.1
```

```{.python .input  n=6}
net = get_net(ctx)
net.hybridize()
train(net, train_data, valid_data, num_epochs, learning_rate, 
      weight_decay, ctx, lr_period, lr_decay)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Train loss: 4.306033, Valid loss 2.901008, Valid acc 0.485054, Time 00:05:11, lr 0.001\nEpoch 1. Train loss: 2.796042, Valid loss 1.284191, Valid acc 0.730978, Time 00:05:16, lr 0.001\nEpoch 2. Train loss: 1.908850, Valid loss 0.823270, Valid acc 0.781250, Time 00:05:15, lr 0.001\nEpoch 3. Train loss: 1.516376, Valid loss 0.691252, Valid acc 0.797554, Time 00:05:14, lr 0.001\nEpoch 4. Train loss: 1.348237, Valid loss 0.585744, Valid acc 0.816576, Time 00:05:16, lr 0.001\nEpoch 5. Train loss: 1.186940, Valid loss 0.556228, Valid acc 0.817935, Time 00:05:17, lr 0.001\nEpoch 6. Train loss: 1.127399, Valid loss 0.524937, Valid acc 0.835598, Time 00:05:18, lr 0.001\nEpoch 7. Train loss: 1.076455, Valid loss 0.498946, Valid acc 0.843750, Time 00:05:18, lr 0.001\nEpoch 8. Train loss: 1.013155, Valid loss 0.487773, Valid acc 0.836957, Time 00:05:18, lr 0.001\nEpoch 9. Train loss: 0.955252, Valid loss 0.491348, Valid acc 0.836957, Time 00:05:17, lr 0.001\nEpoch 10. Train loss: 0.930556, Valid loss 0.489582, Valid acc 0.836957, Time 00:05:19, lr 0.001\nEpoch 11. Train loss: 0.904343, Valid loss 0.477600, Valid acc 0.847826, Time 00:05:16, lr 0.001\nEpoch 12. Train loss: 0.854526, Valid loss 0.453585, Valid acc 0.843750, Time 00:05:18, lr 0.001\nEpoch 13. Train loss: 0.862100, Valid loss 0.458842, Valid acc 0.847826, Time 00:05:18, lr 0.001\nEpoch 14. Train loss: 0.831812, Valid loss 0.439386, Valid acc 0.845109, Time 00:05:15, lr 0.001\nEpoch 15. Train loss: 0.793871, Valid loss 0.456816, Valid acc 0.847826, Time 00:05:15, lr 0.001\nEpoch 16. Train loss: 0.786179, Valid loss 0.434404, Valid acc 0.847826, Time 00:05:15, lr 0.001\nEpoch 17. Train loss: 0.780983, Valid loss 0.442772, Valid acc 0.841033, Time 00:05:20, lr 0.001\nEpoch 18. Train loss: 0.752855, Valid loss 0.458026, Valid acc 0.849185, Time 00:05:18, lr 0.001\nEpoch 19. Train loss: 0.753863, Valid loss 0.450520, Valid acc 0.854620, Time 00:05:16, lr 0.001\nEpoch 20. Train loss: 0.717173, Valid loss 0.447265, Valid acc 0.847826, Time 00:05:21, lr 0.0001\nEpoch 21. Train loss: 0.725081, Valid loss 0.438574, Valid acc 0.860054, Time 00:05:14, lr 0.0001\nEpoch 22. Train loss: 0.690108, Valid loss 0.431900, Valid acc 0.858696, Time 00:05:18, lr 0.0001\nEpoch 23. Train loss: 0.699524, Valid loss 0.430883, Valid acc 0.862772, Time 00:05:14, lr 0.0001\nEpoch 24. Train loss: 0.692978, Valid loss 0.440820, Valid acc 0.861413, Time 00:05:16, lr 0.0001\nEpoch 25. Train loss: 0.700238, Valid loss 0.433091, Valid acc 0.861413, Time 00:05:19, lr 0.0001\nEpoch 26. Train loss: 0.676669, Valid loss 0.429591, Valid acc 0.858696, Time 00:05:18, lr 0.0001\nEpoch 27. Train loss: 0.688726, Valid loss 0.437361, Valid acc 0.857337, Time 00:05:16, lr 0.0001\nEpoch 28. Train loss: 0.681394, Valid loss 0.433691, Valid acc 0.853261, Time 00:05:15, lr 0.0001\nEpoch 29. Train loss: 0.678497, Valid loss 0.425204, Valid acc 0.865489, Time 00:05:10, lr 0.0001\nEpoch 30. Train loss: 0.685778, Valid loss 0.435826, Valid acc 0.860054, Time 00:05:05, lr 0.0001\nEpoch 31. Train loss: 0.665708, Valid loss 0.429318, Valid acc 0.862772, Time 00:05:04, lr 0.0001\nEpoch 32. Train loss: 0.682999, Valid loss 0.431307, Valid acc 0.854620, Time 00:05:07, lr 0.0001\nEpoch 33. Train loss: 0.667949, Valid loss 0.431365, Valid acc 0.860054, Time 00:05:04, lr 0.0001\nEpoch 34. Train loss: 0.683884, Valid loss 0.420280, Valid acc 0.862772, Time 00:05:07, lr 0.0001\nEpoch 35. Train loss: 0.677156, Valid loss 0.425593, Valid acc 0.865489, Time 00:05:09, lr 0.0001\nEpoch 36. Train loss: 0.655331, Valid loss 0.421578, Valid acc 0.862772, Time 00:05:06, lr 0.0001\nEpoch 37. Train loss: 0.667910, Valid loss 0.422135, Valid acc 0.865489, Time 00:05:07, lr 0.0001\nEpoch 38. Train loss: 0.673798, Valid loss 0.428736, Valid acc 0.865489, Time 00:05:06, lr 0.0001\nEpoch 39. Train loss: 0.656416, Valid loss 0.435513, Valid acc 0.862772, Time 00:05:09, lr 0.0001\nEpoch 40. Train loss: 0.666294, Valid loss 0.422588, Valid acc 0.864130, Time 00:05:15, lr 1e-05\nEpoch 41. Train loss: 0.676806, Valid loss 0.438414, Valid acc 0.857337, Time 00:05:07, lr 1e-05\nEpoch 42. Train loss: 0.666304, Valid loss 0.420890, Valid acc 0.869565, Time 00:05:14, lr 1e-05\nEpoch 43. Train loss: 0.684324, Valid loss 0.419047, Valid acc 0.866848, Time 00:05:15, lr 1e-05\nEpoch 44. Train loss: 0.681851, Valid loss 0.428198, Valid acc 0.861413, Time 00:05:12, lr 1e-05\nEpoch 45. Train loss: 0.664063, Valid loss 0.419501, Valid acc 0.865489, Time 00:05:13, lr 1e-05\nEpoch 46. Train loss: 0.667719, Valid loss 0.428660, Valid acc 0.864130, Time 00:05:13, lr 1e-05\nEpoch 47. Train loss: 0.671461, Valid loss 0.430895, Valid acc 0.861413, Time 00:05:10, lr 1e-05\nEpoch 48. Train loss: 0.653425, Valid loss 0.416268, Valid acc 0.868207, Time 00:05:13, lr 1e-05\nEpoch 49. Train loss: 0.659886, Valid loss 0.432814, Valid acc 0.862772, Time 00:05:14, lr 1e-05\nEpoch 50. Train loss: 0.655700, Valid loss 0.418624, Valid acc 0.866848, Time 00:05:16, lr 1e-05\nEpoch 51. Train loss: 0.662815, Valid loss 0.423546, Valid acc 0.860054, Time 00:05:15, lr 1e-05\nEpoch 52. Train loss: 0.667394, Valid loss 0.417992, Valid acc 0.866848, Time 00:05:31, lr 1e-05\nEpoch 53. Train loss: 0.655401, Valid loss 0.426819, Valid acc 0.860054, Time 00:05:41, lr 1e-05\nEpoch 54. Train loss: 0.651320, Valid loss 0.424086, Valid acc 0.853261, Time 00:05:34, lr 1e-05\nEpoch 55. Train loss: 0.659098, Valid loss 0.420166, Valid acc 0.868207, Time 00:05:19, lr 1e-05\nEpoch 56. Train loss: 0.669535, Valid loss 0.432117, Valid acc 0.854620, Time 00:05:30, lr 1e-05\nEpoch 57. Train loss: 0.639944, Valid loss 0.430077, Valid acc 0.855978, Time 00:05:17, lr 1e-05\nEpoch 58. Train loss: 0.663824, Valid loss 0.418473, Valid acc 0.869565, Time 00:05:18, lr 1e-05\nEpoch 59. Train loss: 0.652429, Valid loss 0.429862, Valid acc 0.866848, Time 00:05:19, lr 1e-05\n"
 }
]
```

```{.python .input  n=10}
import matplotlib.pyplot as plt 
%matplotlib inline

plt.plot(loss_train[3:])
plt.plot(loss_valid[2:])
plt.plot(acc_valid)
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "[<matplotlib.lines.Line2D at 0x7f1d1da10e80>]"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "image/png": "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\nCCG80P8D3j1BZrFJudIAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f1d1dae2dd8>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

## 对测试集分类

当得到一组满意的模型设计和参数后，我们使用全部训练数据集（含验证集）重新训练模型，并对测试集分类。

```{.python .input  n=8}
filename = "./mydog_resnet101-v1_train.params"
net.export(filename)
```

```{.python .input  n=7}
import numpy as np

net = get_net(ctx)
net.hybridize()
train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, 
      ctx, lr_period, lr_decay)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Train loss: 4.205145, Time 00:03:42, lr 0.001\nEpoch 1. Train loss: 2.533417, Time 00:03:41, lr 0.001\nEpoch 2. Train loss: 1.745517, Time 00:03:43, lr 0.001\nEpoch 3. Train loss: 1.418851, Time 00:03:41, lr 0.001\nEpoch 4. Train loss: 1.223821, Time 00:03:42, lr 0.001\nEpoch 5. Train loss: 1.128073, Time 00:03:41, lr 0.001\nEpoch 6. Train loss: 1.055831, Time 00:03:44, lr 0.001\nEpoch 7. Train loss: 0.989393, Time 00:03:54, lr 0.001\nEpoch 8. Train loss: 0.965816, Time 00:03:44, lr 0.001\nEpoch 9. Train loss: 0.915763, Time 00:03:52, lr 0.001\nEpoch 10. Train loss: 0.878995, Time 00:04:14, lr 0.001\nEpoch 11. Train loss: 0.845551, Time 00:03:59, lr 0.001\nEpoch 12. Train loss: 0.808450, Time 00:04:08, lr 0.001\nEpoch 13. Train loss: 0.794344, Time 00:04:21, lr 0.001\nEpoch 14. Train loss: 0.791237, Time 00:04:29, lr 0.001\nEpoch 15. Train loss: 0.735471, Time 00:04:03, lr 0.001\nEpoch 16. Train loss: 0.749893, Time 00:04:20, lr 0.001\nEpoch 17. Train loss: 0.726186, Time 00:04:16, lr 0.001\nEpoch 18. Train loss: 0.718196, Time 00:04:27, lr 0.001\nEpoch 19. Train loss: 0.685040, Time 00:04:21, lr 0.001\nEpoch 20. Train loss: 0.666943, Time 00:04:27, lr 0.0001\nEpoch 21. Train loss: 0.666220, Time 00:04:20, lr 0.0001\nEpoch 22. Train loss: 0.652166, Time 00:04:19, lr 0.0001\nEpoch 23. Train loss: 0.658306, Time 00:04:26, lr 0.0001\nEpoch 24. Train loss: 0.659534, Time 00:04:16, lr 0.0001\nEpoch 25. Train loss: 0.654787, Time 00:03:52, lr 0.0001\nEpoch 26. Train loss: 0.666717, Time 00:04:25, lr 0.0001\nEpoch 27. Train loss: 0.647303, Time 00:04:17, lr 0.0001\nEpoch 28. Train loss: 0.651825, Time 00:04:22, lr 0.0001\nEpoch 29. Train loss: 0.624158, Time 00:04:16, lr 0.0001\nEpoch 30. Train loss: 0.644953, Time 00:04:16, lr 0.0001\nEpoch 31. Train loss: 0.660525, Time 00:03:55, lr 0.0001\nEpoch 32. Train loss: 0.637902, Time 00:04:19, lr 0.0001\nEpoch 33. Train loss: 0.648794, Time 00:04:15, lr 0.0001\nEpoch 34. Train loss: 0.655414, Time 00:04:21, lr 0.0001\nEpoch 35. Train loss: 0.640006, Time 00:04:14, lr 0.0001\nEpoch 36. Train loss: 0.621890, Time 00:04:19, lr 0.0001\nEpoch 37. Train loss: 0.629549, Time 00:04:23, lr 0.0001\nEpoch 38. Train loss: 0.618066, Time 00:04:21, lr 0.0001\nEpoch 39. Train loss: 0.637775, Time 00:04:16, lr 0.0001\nEpoch 40. Train loss: 0.613486, Time 00:04:19, lr 1e-05\nEpoch 41. Train loss: 0.628706, Time 00:04:21, lr 1e-05\nEpoch 42. Train loss: 0.608838, Time 00:04:26, lr 1e-05\nEpoch 43. Train loss: 0.620748, Time 00:03:58, lr 1e-05\nEpoch 44. Train loss: 0.621619, Time 00:04:07, lr 1e-05\nEpoch 45. Train loss: 0.611737, Time 00:04:21, lr 1e-05\nEpoch 46. Train loss: 0.609142, Time 00:04:12, lr 1e-05\nEpoch 47. Train loss: 0.621013, Time 00:04:21, lr 1e-05\nEpoch 48. Train loss: 0.623214, Time 00:04:00, lr 1e-05\nEpoch 49. Train loss: 0.620597, Time 00:03:58, lr 1e-05\nEpoch 50. Train loss: 0.616932, Time 00:03:53, lr 1e-05\nEpoch 51. Train loss: 0.612297, Time 00:04:16, lr 1e-05\nEpoch 52. Train loss: 0.647222, Time 00:04:26, lr 1e-05\nEpoch 53. Train loss: 0.620905, Time 00:04:10, lr 1e-05\nEpoch 54. Train loss: 0.623736, Time 00:04:21, lr 1e-05\nEpoch 55. Train loss: 0.620233, Time 00:04:19, lr 1e-05\nEpoch 56. Train loss: 0.622036, Time 00:04:08, lr 1e-05\nEpoch 57. Train loss: 0.618518, Time 00:04:10, lr 1e-05\nEpoch 58. Train loss: 0.611768, Time 00:04:18, lr 1e-05\nEpoch 59. Train loss: 0.622440, Time 00:04:12, lr 1e-05\n"
 }
]
```

```{.python .input  n=8}
filename = "./mydog_resnet101-v1_train_valid.params"
net.export(filename)
```

```{.python .input  n=9}
import matplotlib.pyplot as plt 
%matplotlib inline

plt.plot(loss_train[3:])
plt.plot(loss_valid[2:])
plt.plot(acc_valid)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "[<matplotlib.lines.Line2D at 0x7facd0500710>]"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "image/png": "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\nRK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fac6f8f8eb8>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

```{.python .input  n=10}
import os
outputs = []
for data, label in test_data:
    output = nd.softmax(net(data.as_in_context(ctx)))
    outputs.extend(output.asnumpy())
ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('resnet101-v1.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, outputs):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

上述代码执行完会生成一个`submission.csv`的文件用于在Kaggle上提交。这是Kaggle要求的提交格式。这时我们可以在Kaggle上把对测试集分类的结果提交并查看分类准确率。你需要登录Kaggle网站，打开[120种狗类识别问题](https://www.kaggle.com/c/dog-breed-identification)，并点击下方右侧`Submit Predictions`按钮。

![](../img/kaggle-dog-submit1.png)


请点击下方`Upload Submission File`选择需要提交的预测结果。然后点击下方的`Make Submission`按钮就可以查看结果啦！

![](../img/kaggle-dog-submit2.png)

温馨提醒，目前**Kaggle仅限每个账号一天以内5次提交结果的机会**。所以提交结果前务必三思。


## 作业（[汇报作业和查看其他小伙伴作业](https://discuss.gluon.ai/t/topic/2399)）：

* 使用Kaggle完整数据集，把batch_size和num_epochs分别调大些，可以在Kaggle上拿到什么样的准确率和名次？
* 你还有什么其他办法可以继续改进模型和参数？小伙伴们都期待你的分享。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/2399)
