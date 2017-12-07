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

```{.python .input  n=4}
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
                        brightness=0.125, contrast=0.125, 
                        saturation=0.125, hue=0.125, 
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
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', 
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, 
                                     transform=transform_test)

batch_size = 32
loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, 
                          last_batch='keep')
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

class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

    
class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # 模块1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, 
                              padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            # 模块2
            for _ in range(3):
                net.add(Residual(channels=32))
            # 模块3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))
            # 模块4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))
            # 模块5
            net.add(nn.GlobalAvgPool2D())
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out


def get_net(ctx):
    from mxnet.gluon.model_zoo import vision as models
    pretrained_net = models.resnet50_v1(pretrained=True)
    
    num_outputs = 120
    
    net = models.resnet50_v1(classes=num_outputs)
    net.features = pretrained_net.features        
    net.classifier.initialize(init.Xavier())
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
  "text": "Epoch 0. Train loss: 3.648017, Valid loss 1.595193, Valid acc 0.605978, Time 00:04:34, lr 0.001\nEpoch 1. Train loss: 1.896737, Valid loss 0.997980, Valid acc 0.709239, Time 00:04:38, lr 0.001\nEpoch 2. Train loss: 1.541342, Valid loss 0.891067, Valid acc 0.740489, Time 00:04:35, lr 0.001\nEpoch 3. Train loss: 1.340407, Valid loss 0.843291, Valid acc 0.743207, Time 00:04:36, lr 0.001\nEpoch 4. Train loss: 1.241570, Valid loss 0.797115, Valid acc 0.764946, Time 00:04:37, lr 0.001\nEpoch 5. Train loss: 1.193722, Valid loss 0.751114, Valid acc 0.770380, Time 00:04:41, lr 0.001\nEpoch 6. Train loss: 1.143883, Valid loss 0.796935, Valid acc 0.759511, Time 00:04:38, lr 0.001\nEpoch 7. Train loss: 1.073836, Valid loss 0.725173, Valid acc 0.774457, Time 00:04:41, lr 0.001\nEpoch 8. Train loss: 1.019292, Valid loss 0.697496, Valid acc 0.773098, Time 00:04:44, lr 0.001\nEpoch 9. Train loss: 1.002359, Valid loss 0.733970, Valid acc 0.763587, Time 00:04:44, lr 0.001\nEpoch 10. Train loss: 0.898416, Valid loss 0.685146, Valid acc 0.790761, Time 00:04:41, lr 0.001\nEpoch 11. Train loss: 0.952203, Valid loss 0.732423, Valid acc 0.771739, Time 00:04:39, lr 0.001\nEpoch 12. Train loss: 0.890772, Valid loss 0.750670, Valid acc 0.754076, Time 00:04:36, lr 0.001\nEpoch 13. Train loss: 0.835717, Valid loss 0.743617, Valid acc 0.770380, Time 00:04:36, lr 0.001\nEpoch 14. Train loss: 0.830532, Valid loss 0.730814, Valid acc 0.770380, Time 00:04:38, lr 0.001\nEpoch 15. Train loss: 0.819771, Valid loss 0.709770, Valid acc 0.778533, Time 00:04:43, lr 0.001\nEpoch 16. Train loss: 0.773097, Valid loss 0.728051, Valid acc 0.756793, Time 00:04:41, lr 0.001\nEpoch 17. Train loss: 0.772373, Valid loss 0.741282, Valid acc 0.771739, Time 00:04:41, lr 0.001\nEpoch 18. Train loss: 0.756294, Valid loss 0.783905, Valid acc 0.770380, Time 00:04:39, lr 0.001\nEpoch 19. Train loss: 0.722939, Valid loss 0.734502, Valid acc 0.777174, Time 00:04:40, lr 0.001\nEpoch 20. Train loss: 0.677347, Valid loss 0.685545, Valid acc 0.782609, Time 00:04:40, lr 0.0001\nEpoch 21. Train loss: 0.601153, Valid loss 0.671395, Valid acc 0.783967, Time 00:04:41, lr 0.0001\nEpoch 22. Train loss: 0.573031, Valid loss 0.672643, Valid acc 0.790761, Time 00:04:41, lr 0.0001\nEpoch 23. Train loss: 0.568342, Valid loss 0.663127, Valid acc 0.794837, Time 00:04:44, lr 0.0001\nEpoch 24. Train loss: 0.552932, Valid loss 0.626505, Valid acc 0.794837, Time 00:04:42, lr 0.0001\nEpoch 25. Train loss: 0.551298, Valid loss 0.613697, Valid acc 0.796196, Time 00:04:43, lr 0.0001\nEpoch 26. Train loss: 0.542429, Valid loss 0.636471, Valid acc 0.793478, Time 00:04:42, lr 0.0001\nEpoch 27. Train loss: 0.539684, Valid loss 0.617739, Valid acc 0.800272, Time 00:04:45, lr 0.0001\nEpoch 28. Train loss: 0.526556, Valid loss 0.637061, Valid acc 0.796196, Time 00:04:44, lr 0.0001\nEpoch 29. Train loss: 0.515437, Valid loss 0.625006, Valid acc 0.798913, Time 00:04:41, lr 0.0001\nEpoch 30. Train loss: 0.512611, Valid loss 0.618866, Valid acc 0.793478, Time 00:04:41, lr 0.0001\nEpoch 31. Train loss: 0.527781, Valid loss 0.614290, Valid acc 0.804348, Time 00:04:38, lr 0.0001\nEpoch 32. Train loss: 0.505233, Valid loss 0.614059, Valid acc 0.793478, Time 00:04:37, lr 0.0001\nEpoch 33. Train loss: 0.486404, Valid loss 0.616725, Valid acc 0.796196, Time 00:04:41, lr 0.0001\nEpoch 34. Train loss: 0.519231, Valid loss 0.628522, Valid acc 0.801630, Time 00:04:36, lr 0.0001\nEpoch 35. Train loss: 0.500089, Valid loss 0.620646, Valid acc 0.796196, Time 00:04:40, lr 0.0001\nEpoch 36. Train loss: 0.510873, Valid loss 0.622715, Valid acc 0.808424, Time 00:04:42, lr 0.0001\nEpoch 37. Train loss: 0.511445, Valid loss 0.631808, Valid acc 0.792120, Time 00:04:55, lr 0.0001\nEpoch 38. Train loss: 0.495401, Valid loss 0.615686, Valid acc 0.796196, Time 00:04:48, lr 0.0001\nEpoch 39. Train loss: 0.493837, Valid loss 0.616575, Valid acc 0.798913, Time 00:05:28, lr 0.0001\nEpoch 40. Train loss: 0.494594, Valid loss 0.620549, Valid acc 0.796196, Time 00:05:18, lr 1e-05\nEpoch 41. Train loss: 0.499696, Valid loss 0.616849, Valid acc 0.797554, Time 00:04:59, lr 1e-05\nEpoch 42. Train loss: 0.512922, Valid loss 0.622015, Valid acc 0.800272, Time 00:05:35, lr 1e-05\nEpoch 43. Train loss: 0.487422, Valid loss 0.615936, Valid acc 0.798913, Time 00:05:26, lr 1e-05\nEpoch 44. Train loss: 0.479388, Valid loss 0.612243, Valid acc 0.796196, Time 00:05:09, lr 1e-05\nEpoch 45. Train loss: 0.489271, Valid loss 0.627334, Valid acc 0.793478, Time 00:05:03, lr 1e-05\nEpoch 46. Train loss: 0.470111, Valid loss 0.613673, Valid acc 0.804348, Time 00:04:49, lr 1e-05\nEpoch 47. Train loss: 0.484191, Valid loss 0.610538, Valid acc 0.796196, Time 00:05:30, lr 1e-05\nEpoch 48. Train loss: 0.479520, Valid loss 0.612305, Valid acc 0.801630, Time 00:05:09, lr 1e-05\nEpoch 49. Train loss: 0.467191, Valid loss 0.605321, Valid acc 0.796196, Time 00:04:57, lr 1e-05\nEpoch 50. Train loss: 0.472137, Valid loss 0.614814, Valid acc 0.807065, Time 00:05:08, lr 1e-05\nEpoch 51. Train loss: 0.487789, Valid loss 0.617371, Valid acc 0.802989, Time 00:05:12, lr 1e-05\nEpoch 52. Train loss: 0.472996, Valid loss 0.612178, Valid acc 0.792120, Time 00:05:25, lr 1e-05\nEpoch 53. Train loss: 0.484471, Valid loss 0.616479, Valid acc 0.797554, Time 00:05:31, lr 1e-05\nEpoch 54. Train loss: 0.482880, Valid loss 0.610697, Valid acc 0.796196, Time 00:05:27, lr 1e-05\nEpoch 55. Train loss: 0.473708, Valid loss 0.614730, Valid acc 0.804348, Time 00:05:00, lr 1e-05\nEpoch 56. Train loss: 0.476842, Valid loss 0.606947, Valid acc 0.813859, Time 00:04:51, lr 1e-05\nEpoch 57. Train loss: 0.466122, Valid loss 0.613744, Valid acc 0.802989, Time 00:04:52, lr 1e-05\nEpoch 58. Train loss: 0.479256, Valid loss 0.600857, Valid acc 0.801630, Time 00:05:28, lr 1e-05\nEpoch 59. Train loss: 0.476287, Valid loss 0.607429, Valid acc 0.797554, Time 00:05:01, lr 1e-05\n"
 }
]
```

```{.python .input  n=7}
import matplotlib.pyplot as plt 
%matplotlib inline

plt.plot(loss_train[2:])
plt.plot(loss_valid)
plt.plot(acc_valid)
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "[<matplotlib.lines.Line2D at 0x7fc40f99bd68>]"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXcAAAD8CAYAAACMwORRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xd4XMW9xvHvrFZa9V5sy5LlJrmAe8fGxjbYBgwkIfRy\nE2qABHJDCQmXkHBzQwI3gZuAQwkhlEASMLHpGPdu3Ksk27JlSbbVe90y94+RhWSr2iutdv37PI+e\nlXaPzplRec/snDkzSmuNEEII32LxdAGEEEK4n4S7EEL4IAl3IYTwQRLuQgjhgyTchRDCB0m4CyGE\nD5JwF0IIHyThLoQQPkjCXQghfJDVUweOjY3VKSkpnjq8EEJ4pW3bthVpreM62s5j4Z6SksLWrVs9\ndXghhPBKSqnszmwn3TJCCOGDJNyFEMIHSbgLIYQPknAXQggf1GG4K6VeV0oVKKX2trPNLKXUTqXU\nPqXUavcWUQghRFd1puX+BjC/rReVUpHAS8BVWuuRwHfdUzQhhBBnq8Nw11qvAUra2eQmYLHW+ljj\n9gVuKpsQQoiz5I4+91QgSim1Sim1TSl1W1sbKqXuVkptVUptLSwsPLuj5e+H5U9DdfFZFlcIIXyf\nO8LdCowHrgDmAf+llEptbUOt9Sta6wla6wlxcR3eYNW64oOw9jmoPHG25RVCCJ/njjtUc4FirXU1\nUK2UWgOMBjLdsO8zBYSax4aqbtm9EEL4Ane03JcA05VSVqVUMDAZOOCG/bbOFmYeJdyFEKJNHbbc\nlVLvArOAWKVULvALwB9Aa/1nrfUBpdTnwG7ABbymtW5z2OQ5Cwgxj/US7kII0ZYOw11rfWMntnkW\neNYtJeqIdMsIIUSHvO8O1aZumWrPlkMIIXox7wv3pm6ZSs+WQwghejHvC3erDSz+0nIXQoh2eGyx\njrOltUb7h0B9pReemYQQomd4XT5+uCOP43VWqivLPF0UIYTotbwu3AfEhFCtA6mtqvB0UYQQotfy\nunAfFBtCDYE01Ei4CyFEW7wu3KNCAqizBOGsk9EyQgjRFq8LdwACwlB2GS0jhBBt8cpwtwaFYpVw\nF0KINnlluNuCI7DpWmobnJ4uihBC9EpeGe5BoeGEUkd2ibTehRCiNV4Z7uHhUdiUnaP5MtZdCCFa\n45XhHhkVDUBeQZGHSyKEEL2TV4a7LdjMDHlSwl0IIVrlleF+ak73ohJZJFsIIVrjneHeOKd7SVmJ\nhwsihBC9k3eGe+Oc7o7aSirq7B4ujBBC9D5eGu6mWyaEOo4WyXBIIYQ4nXeGu+2bcD8i4S6EEGfw\nznBvbLmHKgl3IYRojVeHe58gp4S7EEK0wjvD3T8IlIXEYIf0uQshRCu8M9yVgoBQ4gPsZBVVo7X2\ndImEEKJX8c5wBwgIJSbAQWWdg5LqBk+XRgghehUvDvcQIv3qAaTfXQghTtNhuCulXldKFSil9naw\n3USllEMpda37itcOWyhhljpAwl0IIU7XmZb7G8D89jZQSvkBvwW+dEOZOicglCBdi9WiJNyFEOI0\nHYa71noN0NEkLj8EPgAK3FGoTgkIRTVUkRwdLOEuhBCnOec+d6VUIvAtYFEntr1bKbVVKbW1sLDw\n3A5sC4WGalJiQyTchRDiNO64oPo88JjW2tXRhlrrV7TWE7TWE+Li4s7tqAEhUF/FwNgQjhZX43LJ\ncEghhDjF6oZ9TADeU0oBxAKXK6UcWut/u2HfbQsIhQYT7nV2F/mVdfSNCOrWQwohhLc453DXWg88\n9blS6g3g424PdjBzuttrGBgdCMCRwmoJdyGEaNSZoZDvAhuBNKVUrlLqDqXUvUqpe7u/eO1onNN9\nUIQC4Eix9LsLIcQpHbbctdY3dnZnWuv/OKfSdEXj5GEJNjs2q4UjhRLuQghxivfeodq41J7FUdN0\nUVUIIYThveHe2C1DfSUpMSFkyXBIIYRo4sXhbrplaKhmYFwIx4prcDg7HI0phBDnBS8O98aWe0MV\nA2NCcLg0eWW1ni2TEEL0Et4b7o197tRXMTDOBP3hwioPFkgIIXoP7w33pm6ZKkb0DSfAamH9oWLP\nlkkIIXoJLw73b7plQmxWpg6KYUV6z81bJoQQvZkXh3tjy73edMXMGR7PkaJqsqRrRgghvDjcLRbw\nD4EGE+azh8UDsPyAtN6FEMJ7wx1M10xjuPePCmZYnzCWp+d7uFBCCOF53h3uttCmbhkwrfevj5ZS\nXmv3YKGEEMLzvDvcA8yCHafMGR6P06VZk3mOC4EIIYSX84Fw/6blPiYpiuiQAJYfkK4ZIcT5zbvD\n3dYy3P0sillpcazKLJSpCIQQ5zXvDvfGpfaamzMsgbIaOztyyjxUKCGE8DwvD/eWLXeAGamxWC2K\nr6RrRghxHvPucLeFtbigChAe6M/kQdGskPHuQojzmHeH+6lx7lq3eHr2sAQOFlRxrLjGQwUTQgjP\n8vJwDwXtAnvLEJ9z6m5VuaFJCHGe8u5wt32zYEdzKbEhDI4LkYnEhBDnLe8O96bJwyrPeGnO8AQ2\nZRVTVe/o4UIJIYTn+Ua4N5y5fursYfHYnXK3qhDi/OTl4f7NnO6nmzAgin4Rgfzf8oNyQ5MQ4rzj\n3eHebKm901n9LDy5cCTpJyt5Y8PRni2XEEJ4mHeHe7Ol9lozb2QCs4fF8/tlmRyXxbOFEOcRLw/3\ntrtlAJRS/PKqkbi05lcf7e/BggkhhGd1GO5KqdeVUgVKqb1tvH6zUmq3UmqPUmqDUmq0+4vZBlvL\npfZakxQdzI/mDOXzfSdZIePehRDnic603N8A5rfz+hFgptb6QuBp4BU3lKtz2hkt09yd0wcxND6U\nJ5fso7bB2QMFE0IIz+ow3LXWa4CSdl7foLUubfxyE9DfTWXrmJ8/+Nmg4cxx7s0FWC389zUXkFta\nyx9XHOyhwgkhhOe4u8/9DuCztl5USt2tlNqqlNpaWOim8ee20A5b7gCTB8Vw7fj+vLo2i4P57Z8M\nhBDC27kt3JVSl2DC/bG2ttFav6K1nqC1nhAXF+eeA7cyp3tbHl8wjBCblf9ashd92mRjQgjhS9wS\n7kqpUcBrwNVa62J37LPTAsLaHC1zuphQGw9flsamrBI+2XOimwsmhBCec87hrpRKBhYDt2qtM8+9\nSF1kO3PBjvbcOCmZkf3C+fUnB6hpkHlnhBC+qTNDId8FNgJpSqlcpdQdSql7lVL3Nm7yJBADvKSU\n2qmU2tqN5T1TF7plwKyz+qurR3KivI4XVx7qxoIJIYTnWDvaQGt9Ywev3wnc6bYSdVVAKFQc79K3\njB8QzbfHJvLqmiN8d3wSKbEh3VQ4IYTwDO++QxVaXWqvM366YBgBVgu/+ljuXBVC+B7vD/eAkFbn\nc+9IfHggD84Zyor0ApbLYtpCCB/jA+Ee2uo6qp1x+7QUBseF8KuP91NnlztXhRC+w/vD3RYKLgc4\nG7r8rQFWC09dNZLs4hr+su5INxROCCE8w/vDPaDjycPaM2NoHJeOSODl1Ydl3hkhhM/wnXDvwlj3\n0901YxAVdQ6W7MxzU6GEEMKzfCDc25/TvTMmpkQxrE8Yf9uYLdMSCCF8gveHeyfmdO+IUorbpqZw\n4EQF27JLO/4GIYTo5bw/3AMa11E9h5Y7wDVj+xEWaOVvG7PdUCghhPAsHwj3c++WAQgOsPLd8Ul8\ntucEBRV1biiYEEJ4jveHuxu6ZU65deoAHC7Nu1tyznlfQgjhSd4f7k3dMl2fguB0A2NDmJkaxzub\ns7E7Xee8PyGE8BQfCPdT3TLuWV3ptqkDKKis54t9J92yPyGE8ATvD3erDSxWt7TcAWalxZMUHcSb\ncmFVCOHFvD/clerynO7t8bMobpk8gC1HSjhwosIt+xRCiJ7m/eEOXVpqrzOum5CEzWqR1rsQwmv5\nRrh3cam9jkSFBHD1mH58uCOXfBkWKYTwQr4R7m7sljnl/kuG4HBqXlh+0K37FUKInuAj4e7eljvA\ngJgQbp6czD++zuFwoXv3LYQQ3c03wv0sl9rryA/nDCXQauG5LzLcvm8hhOhOvhHuZ7nUXkdiQ23c\nffFgPtt7ku3HZEIxIYT38JFwd3+3zCl3zhhIbGgAz3yaLtMBCyG8ho+Ee0i3dMsAhNisPDhnKFuO\nlrAyo6BbjiGEEO7mG+FuCwNHHTgd3bL7GyYlkxITzG8/y8Dpkta7EKL3841wd8NSe+3x97PwyLxh\nZORXsnh7brccQwgh3MlHwt09c7q35/IL+zC6fwS/X5bJwXz3X7wVQgh36jDclVKvK6UKlFJ723hd\nKaX+Tyl1SCm1Wyk1zv3F7IAb53Rvi1KKX1w1kup6B/NfWMsvluylrKah244nhBDnojMt9zeA+e28\nvgAY2vhxN7Do3IvVRW6c070945KjWPXIJdw0KZm3NmUz89lVvLH+iMz9LoTodToMd631GqCknU2u\nBt7UxiYgUinV110F7BQ3z+nenuiQAJ6+5gI+fXAGFySG89RH+7n8hbUUVtZ3+7GFEKKz3NHnngg0\nX5cut/G5Myil7lZKbVVKbS0sLHTDoRud6papK3ffPjswrE84b98xmZduHsfBgiqW7MzrsWMLIURH\nevSCqtb6Fa31BK31hLi4OPftOGYo+AVAzhb37bMTlFJcfmFfhsSHsjrTjScrIYQ4R+4I9zwgqdnX\n/Ruf6zkBwZA8FQ6v7NHDnjIzNY7NR0qobXB65PhCCHE6d4T7UuC2xlEzU4ByrfUJN+y3a4bMgYJ9\nUNHzh56VFkeDw8WmrOIeP7YQQrSmM0Mh3wU2AmlKqVyl1B1KqXuVUvc2bvIpkAUcAl4F7uu20rZn\n8GzzmNXzrfeJKdEE+fuxSqYnEEL0EtaONtBa39jB6xq4320lOlsJF0BIPBxaDmNu6tFDB/r7MXVw\njPS7CyF6Dd+4QxXMQtmDZ5uWu6vnx53PTI3jaHENR4u6d6y9EEJ0hu+EO5h+95piOLmrxw89K82M\n/pGuGSFEb+Bb4T7oEvN4aHmPH3pATAgpMcFtds04nC5u/ctmXllzuIdLJoQ4H/lWuIfGQZ9RHhsS\nOSstno1ZxdTZzxwS+cH2XNYeLOL/lh+ivMbugdIJIc4nvhXuYPrdczZ1y7J7HZmZGked3cWWIy1n\na6hpcPD7ZZmkxARTVe/gbxuP9njZhBDnF98L9yFzwOWAo+t6/NBTBsUQYLWwKqNl18zr646QX1HP\nc98dzexh8by+/gjV9d2zsIgQQoAvhnvSZPAP9ki/e1CAH5MHRrM685uLqkVV9fx5dRbzRiYwISWa\n+y8ZQlmNnXe3HOvx8gkhzh++F+5WG6TMgMMrPHL4WWnxHC6sJqekBoD/W36QWruTR+cPA2D8gCim\nDorhlTVZrfbNCyGEO/heuIPpdy85DKVHe/zQTUMiMwvJKqzi75uPceOkJAbHhTZt88DsIRRU1vP+\nNlmyTwjRPXwz3IfMMY8e6JoZFBtC/6ggVmcU8uwXGdisFh6ck9pim2mDYxidFMmfVx+WhT6EEN3C\nN8M9ZghEJHmka0Ypxay0OFZnFvDZ3pPcM3MwcWG2M7Z54JIh5JbWsnTn8R4voxDC9/lmuJ+aiuDI\nGnD2/Jjymanx2J2auDAbd84Y2Oo2c4bFM6xPGC+tOoTLpXu4hEIIX+eb4Q4m3OsrIG9bjx962uAY\nBsWF8MQVwwkOaH1uNotFcd8lQzhcWM0X+072cAmFEL7Od8N90EywWGHHWz1+6BCblRU/mcXVY1pd\nbbDJFRf2ZWBsCC8sPyitdyGEW/luuAdFwZQfwI634dhmT5emVX4WxUNzh5J+spKlu6TvXQjhPr4b\n7gAzfwrh/eHjH3uk770zFo7qx4i+4fzvsgwaHDJyRgjhHr4d7rZQuPx3Zvm9TYs8XZpWWSyKR+en\nkVNSy983Z3u6OEIIH+Hb4Q4w7ApIuxxW/QbKcjxdmlbNTI1jyqBo/rjiEFUy54wQwg18P9wBFvzW\nPH72mGfL0QalFI/NH0ZxdQN/WXvE08URQviA8yPcI5Nh5mOQ8Qmkf+Lp0rRqbHIU80f24ZU1hymu\nqvd0cYQQXu78CHeAqfdD3HD49FGor/J0aVr18Lw0au1O/rTykKeLIoTwcudPuPv5w5V/gIpc0//e\nCw2JD+W6CUm8s+lY06ySQghxNs6fcAcYMBXG3QabXoLjOz1dmlY9NDcVpeBnH+7h870nOFJUjVNu\ncBJCdJHS2jPBMWHCBL1169aeP3BtKfxpEoT1gbtWgl/r0wN40surD/PM5+mc+tUE+ltITQjjgsQI\nrhzVlykDY7BYlGcLKYTwCKXUNq31hA63O+/CHWDfv+Fft8OlT8NFP/JMGTpQ0+DgYH4VGScrST9Z\nSUZ+Bbtyyqmqd5AYGcTVY/rx7XGJDIkP83RRhRA9SMK9PVrDezfB4ZVw30aIbn3mxt6mtsHJsgP5\nLN6ey9qDRThdmrHJkSy6eTx9IgI9XTwhRA/obLh3qs9dKTVfKZWhlDqklPppK69HKKU+UkrtUkrt\nU0p972wK3WOUgsufMxOLffwQeOgE11VBAX5cNbofb3xvEhsfn80TVwznwIkKnlyy19NFE8Lnaa05\nWHqQRbsW8fjax/ny6Jc0OBs8Xaw2ddjhrJTyA14ELgVyga+VUku11vubbXY/sF9rvVApFQdkKKXe\n0Vr33ppHJMLcX8CnD8Ou92DMjZ4uUZfEhwVy54xBOFyaZz5L5/O9J5l/QR9PF+u8oLVGqe675pFV\nnsW/Mv5FjaOGOy+4k6TwpG47Vk9xaRcW5Z7xGy7tYsvJLewq2EVCSAJJYUkkhyUTGxTr9t+L1pr9\nxfv56thXfJX9FUcrjqJQhNvC+TjrY8IDwlkwcAELBy9kVOyobv276KrOXE2cBBzSWmcBKKXeA64G\nmoe7BsKUqVkoUAL0/vvoJ9wBe/4FXzwOQ+ZCaJynS9Rld0wfyJKdx3lq6T4uGhJDWKC/p4vUa5TU\nlbC3aC+h/qEkhSW1+c+vtabaXo3VYiXQ2nr3VmldKZ8e+ZSPDn9EZmkms5JmsXDQQqb3n46/pf2f\nucPlILsim4ySDA6WHSTUP5Rh0cNIi04jNii2aZtVOat4L/09Np/cjNVixd/iz9LDS7kh7QbuGXUP\nkYGRnaq30+WkoqGCUP9Q/P069/dgd9kpqS2huK6YotoiGpwNBFuDCfYPJsgaRLB/MGH+YYTbwjsd\n0g6Xg00nNrH00FJW5KwgLiiOaf2mMS1xGpP7TCY0wKwrrLXmePVxMkoyyCjNQGtNalQqadFpJIYm\nNh0vqzyLjw5/xMdZH3Oy+sw1EIKsQSSHJTM7eTYLBy0865NiaV0pG49vZP3x9Ww8vpHC2kL8lB+T\n+kzi1hG3Mjt5NlG2KFO3w0tZcmgJ/8j4B4mhicQHx7f8uVmD6Rval6SwpKaPEP+QsypXV3XY566U\nuhaYr7W+s/HrW4HJWusHmm0TBiwFhgFhwPVa6zNuBVVK3Q3cDZCcnDw+O7sXTJRVkA5/ng4jr4Hv\nvNb+tid2w4Y/wsIXICC4Z8rXCTtzyvjWS+u5bcoAfnn1BZ4uTpfV2GvIrcolpzKHivqKM153aAc1\n9hpqHDXU2mupcdTgcDmICowiNiiWmMAYYoJiCPEP4UDxAXYU7GBn4U6yK1r+fQVZg+gf1p+kUPNP\nX1RXRHFtMcW1xdQ567AoCwPCB5AWlUZadBqpUak0OBtYengpa3PX4tAOhkUPY2TMSFbmrKSkroTo\nwGgWDFzAZQMuw6VdLfZZUFPAobJDHCo7RL3T3HXsp/xwamdTmaIDo0mLSuNw+WEKagroE9KH61Kv\n41tDv4XWmhd3vsiHhz4kxBrC3aPu5sbhN1LVUEVOZU7TR15VHoU1hU3BXFZfhkubGUbDA8LNzygo\nhpjAGDT6m5+lo5Yaew3l9eWU1pd26ndlVVaiA6PN/oJiiA2KbfodnDqOv8Wf5ceW83HWxxTVFhFh\ni2Bu8lyKa4vZfHIztY5arMrKqDjT0s0szaSyobLpGAqFxuRSiH8IqVGp2J129hbvxU/5Ma3fNK4a\nfBXTE6dTUlfCscpj5FTmcKziGOkl6WzL34ZGMy5+HAsHL2ReyjwcLgcZpRlklGSQWZpJZmkmdqed\nYP9ggq3BBPkHEWQN4ljFMfYX70ejibBFMLXvVKYnTmdW0iwibBGt/kyqGqpYlr2MNblrqLRXNv2N\n1thrqLRXtqjbqd/57SNv5/sXfL9TP/PTue2CaifD/VrgIuA/gcHAMmC01vrM/9RGHr2gerpVz5gb\nm25+H4Ze2vo2jgZ4ZSYU7Ier/gTjbj3nw2qteS/jPWodtdw6/NZOt7Ja89TSffxt41E++ME0xiVH\nnXPZuoPWmpzKnKbwzSrL4ljlMYpqizq9D5ufjWBrMH4WP8rqynDoM98gRtoiGRM/hrHxY7kw9kLq\nnfVN//y5leYkYrFYvgmkxpNDjaOm6Z8/ryqvaX+xQbFcMfAKFg5eSFp0GmBauhvyNrDk8BJW5azC\n7mo5pbSf8iMmMIaBkQObThZpUWkMihjUdJzmYRMTFMN1qddxcf+L8bP4tdjXwdKD/H7b71mXtw6r\nsraos0IRHxxPfHB8U4DHBsUSaYukyl5lTjR15mRTVFuERVmaWuLBVtO6jLRFNgX1qZ9FoDWwKfxP\nnQgq6ita7OvUyaSktuSM34NVWZnRfwZXD76aGf1nEOAXYH5uTjs7C3eyPm89m05sws/iR1pUGsOi\nh5EalUpqVCpKKQ6VHmr6+WSUZuBwOZiXMo8rBl3R9G6nLSeqTvDJkU9YcmgJRyuOnnFCjQuKIzUq\nlWD/4Bb1q7ZXExcUx9R+U7mo30WMiBlxxu/ibFQ2VLY4GedW5jKl3xTmp8w/q/25M9ynAk9prec1\nfv04gNb6N822+QR4Rmu9tvHrFcBPtdZb2tpvrwp3Rz28fDE0VMN9m8xUwadb/TtY+WuzCEjkALhn\n9Tkd0qVdPPv1s7x94G0AhkQO4ZfTfsmouFFntb+qegeX/n41EUH+fPTD6fj7ufH+tKoCs1xh2oJ2\nN6toqCCrLIuKhooWwVDVUEV6STo7CnZQXFcMQJh/GKnRqU39pafeskYFRqFo2XViUZamt7lWyzc9\niS7toqK+gqLaIorqiqior2Bo1FBSwlPOue+zsqGSzNJMHC4H4xPGtzju6crry9l6civB/sFNLdkI\nW4Tb+phP2Xh8I+vz1tMnpA/J4cn0D+tP/9D+TcHpKc1/D8V1xVQ2VDI+YTxRgZ5tZGit2Vu0l+XH\nlhMVGNXU1RMdGO3Rcp0rd4a7FcgE5gB5wNfATVrrfc22WQTka62fUkolANsxLfc2m2S9KtwBcrbA\nXy6DyffCgmdavlaYYbpuhi+E5KnmIuxdKyBx/Fkdyu6y84v1v+CjrI+4efjNTO07lac3PU1BTQE3\nD7+ZH479IcH+Xe/2WbY/n7ve3Mqj89O4b9YQwPzjHSg+wLb8bYzvM56RMSO7tlOnA16fB3lb4eYP\nYOhcAApqCthTuIf00nTSS9LJLMnkeHXbq0klhiYyLn5cU4t6cORgt4efEOeDzoZ7hxdUtdYOpdQD\nwBeAH/C61nqfUurextf/DDwNvKGU2gMo4LH2gr1XSpoEk+6CzX+GC76DPXEMDpeDIIsNlv4QAkJg\n/m/BaoNlv4CvX8fedxQfHf6IxQcXo9HmQkrjxZQQ/xBGx41mVtIswgK+udGozlHHI6sfYVXuKh4Y\n8wB3j7obpRTjE8bz/PbnefvA26zMWcmTU55kWuK0DotdVldGg8sMShqdopg90sYLqzbxddGXFDp2\nc9K+h3qX6R3zU348OO5Bbh95e5vBmlOZQ3pJetPFoOBd/yI4fye1odHsWv4Tth+fx86i3U3dFqf6\nqUfFjeK7ad9laORQogKjzrgYZ/OznetvSAjRBefnTUynqXPUsadoDxkFu8jY+Acy/K0c8vdDo5kV\nMoCFhzYxY+5z+I+7BQDH0h/xyaEl/Dl5GLnVx0mNSiUm0PTZnrqQUlFfQaW9EqvFypS+U5ibPJdJ\nfSbxxPon2FGwg59P/jnXD7v+jLJsy9/GUxue4mjFUS4bcBmPTHyEPiFnDnE8VnGM57c/z7LsZW3W\nSztCcVQPxVGVirM2icD4L7CG72FC/FSem/UbYoJimrbNrczlld2vsPTw0hb9k6eL9QtibP/pjIkb\nw5j4MQyNGkqQNagrP24hxDmQO1Q7aU/hHh5Z80hTSzTaP5S0ikKGJYzHHpfKZ5mLKfazEGWLYsHA\nBQyOHMybu18lu+Ykw22xPDD9l8xInHFGH69Lu9hduJuvsr/iq2NfNe3fqqz8z4z/YcHAtvuv6531\nvLH3DV7b8xpKKe4ZdQ+3jbgNfz9/yurKeHn3y7yX8R7+Fn9uGnYTiWGJLb7fgoWRsSPNxSkUTpem\nrNbOK2sO8+a+97DGfkSwNYxnZvyGYbEDeWX3Kyw5tASLsnBd2nVcOfhK7PVV1Cy5j1pHDTVzn8Ri\nC2fU5r/SP2s96v4tEOn9Y6+F8EYS7h1waRdv7nuTF7a/QHxwPI9OfJTR8aPNlfj3vw8HPoK+Y3Dk\n72XDd/7I0vxNrDy2kgZXA2lRadxXmM8lNTWoB7aaO17bobUmvSSddXnrGJcwjvEJneurz6vK43db\nfseKnBWkhKcwd8Bc/pH+D6od1XxryLe4f8z9xAV3bWx+XlktT32+jA0Vz2MJKMLP4odFKa5NvZY7\nLriDhJAEs+Hnj5vZM29ZDEPmmOdKs+HFSZA6H677W5eOK4RwDwn3dpTUlfDzdT9nXd465ibP5alp\nT7Ucw1pVCC9ONDNIzvsNTL0PMKNBssuzGRk7Esvuf8KH98BtS2HQzG4t79rctTyz/kmO1RUxI3Qg\n/znuQYakzO7wpNKe7Tn5PPjlf1NQWc+vZj7A9WObjdLJWg1vXgUT74Irnmv5jat+C6v+B279Nwy+\n5KyPL4Q4OxLuzdQ6apvG5uZU5vD8tucpqy/j0YmPcl3ada0Pmzu4DDK/MOuvtjbW1V4Hvx8OA2fA\ndW92X+ErT8Lnj9OwbzEFAUH0b6g1z4fEQfIUGDAdLrwWQtof+9ua2gYnN766iQMnKvj7XVMYPyAK\nastg0UW3thmyAAAbOUlEQVTgHwj3rD3zZi17Hbw0GfwC4N71YPXsMDwhzjfnfbgX1Rbx8OqHSS9J\np9pe3eK1lPAUnpv5XNMNKWftyydg40vw430Q3vfc9nU6lxO2vg7Lf2XG4V/8MEz7EZTnQPYGOLYR\nstdD2THws8Go62DKDyCha0Mdi6vq+c6iDVTX1PLJnELid78MhelwxzLo30b3Ucbn8O71cOmv4KIH\n3VBZIURnndfhXl5fzve/+D45lTl8e+i3z7g9ekjkEPfc+FF8GP44Dmb9DGY91v62LheseNq0tlPn\ntb9tYabp8jm+HQbNgit+DzGDW9+24ABsftlMfuaohZQZMOU+0y9u6cQ48toySta+gn3DIhIowRGd\ninXuEzDi6va/753r4Og6+OFWCO/X8XGEEG5x3oZ7jb2Gu5bdxYHiA7w450Wm9pvq9mO08Na3zPw0\nD+1pf1WnFb+GNb+DwEj44ba2u1EcDfDyDHNX6OXPwgXf6Vzfek0JbH8Ttrxq1okddAlcs6jtdxQN\n1bDmWbN9QxWVfafxk9wZ5MdP5917phEc0MEtECVZ8OIUGH4lXPt6x+UTQriFW+dz9xb1znp+tOJH\n7Cvax7Mzn+3+YAeYeCdUHodl/9X2vPD7l5pgHzoPGqrgq1+0vb+NfzTdItcsMn3pnb1oGhwN0x+C\nB3eZln7OZlg0DdI/PXPb9E/hxcmw7g9mSoF71hJ2z2d898Y72HO8kgff20mHJ/3oQTD9x7D3A3MB\nVgjRq/hMuNtddh5e/TCbT27m6YueZk7ynJ45cNrlZsqCTS+ZO1ldp90AlL8fPrwXEifA9W+ZLpMd\nb5vpDk5XkmXmsBl+FaSd3aRC+Flh4h1w92qI6A/v3Qgf/xgaaqA8F9672TxnC4Pvf2FmwuxrRspc\nOiKBxxcMZ9n+fD7ckdfBgTAnk8gB8Okj5h2HEKLX8Ilw11rzxLonWJWzip9P/jkLBy/suYMrBfOf\ngZmPwY63zBj5U0FXU/JNkF7/tpm6YOZjENYPPvlPM2/LN5WATx4Gi78ZoXOu4lLhzq9g2g/NhdlF\n00xr/dBymPsU3LPG9P+f5vvTBzIuOZJffbyfoqr69o/hHwQLfgdFGbB50bmXWQjhNj4R7lvzt/Lp\nkU+5b8x93DDshp4vgFJwyc/gsl/D/n+bQK+vNEFfcdwE+6m+b1sozP8fOLnHhO4pez+Aw8thzn+5\n7wKl1QaX/bcZk+5ywICL4P5NpjuljemF/SyK335nFDX1Tp5auq/VbVpImw+pC8z49/JOtPaFED3C\nJ8J9bd5arBYrt424zbMFmfYAXPVHOLwCXhgDWStN/3fSxJbbjbjGXPBc8d/mwmltmbkjtN9Y04fv\nboMvMRd8b/4nRKV0uPnQhDAemD2Ej3efYNn+/I73v+AZ0E748ufnXlYhhFv4RLivy1vHuPhxPbZ8\nVbvG3WZGj9SVm7741hb1OLVAt70Glj0Jy38JNUVw5fOt3zDlDl28m/XemYNJSwjjiX/voaLO3v7G\nUSkw4yew70M4vPLsyyiEcBuvD/eT1Sc5WHqQ6YnTPV2Ub4z8Fjx62PTFtyV2CFz0I9j1rumemfwD\n6Dem58rYgQCrhd9eO4rCynqe+Sy942+Y9iOIGmgurtrrur+AQoh2eX24r89bD9C7wh0gMKLj1vKM\nhyEiGcITTZ99LzMmKZLvXzSQv28+xqas4vY39g804/KLD8KfJpjx8/banimoEOIMXh/u6/LWkRCc\nwJDIIZ4uStcFBJsRLXetbH1pv17gPy9LJTk6mMcX78HhdLW/8dBLzSyS4YlmtarnR8H6F8zFZSFE\nj+pwJabezO6ys+nEJualzDvnNTM9JizB0yVoV3CAlccXDOMH72xndWYhc4Z3UN4hc2DwbDPvzZrn\nzDWFtb+HyGRzjaGhBuzVplUfNdAsXTh8IfQdfU6zXAohWvLqcN9ZsJMqexUzEmd4uig+be6IBGJD\nA/jH1zkdhzuYkE6Zbj5yt8GWl6GuwrxT8Q8C/xDTjXN8h7lLdu1zJvyHXwVjb4X4Yd1fKSF8nFeH\n+7q8dViVlcl9J3u6KD7N38/Ct8f15/V1RyisrCcurAvrofYfD/1fafv16mLI+BQOLDUToG1/0yw+\nHjv03AsuxHnMq/vc1+WtY2zCWEIDemd/tS+5bkISDpdm8fZc9+44JMYMF735X2ZCNb8AeO8m09IX\nQpw1rw33/Op8Mksze98oGR81JD6U8QOi+MfWnI4nFTtbUQPM8n3Fh82Ux64OLuAKIdrkteG+/ngv\nHQLpw66fkERWYTXbsku77yAp0839ARmfwup27hMQQrTLa8N9Xd464oPjGRopfbM95YpRfQkJ8OMf\nX+d074Em3QVjboHVvzULlXdGaba5gSpve/vbVeabWTI/fxwKM869rEL0Ul4Z7naXnY3HNzIjcYb3\nDoH0QiE2K1eO6scne05QVe/o+BvOllJwxf9C4ngzXXLBgfa3z/wSXr4YtrwCr82Bz39mFiNpTmuz\nWtWLk8yUy1teNZ+/cSXsXdxzUxYXpJupl7uivhL2/RsW3w1/W2jW9/XQIjvCe3jlaJldBbuosldJ\nl4wHXDcxiX9szeHjXce5YVJy9x3IP9DMpvnyTLPa1UUPwdibzfTJp7icsOoZsxBKwoVw2xLY/jfY\n9KJp8V/5Bxg618xW+fFDcPBLSJoMV79oVsTa8RZs+yu8/z0IiYeR15h59xPHQfTgzi1T2Bn1lbDn\nfXOsE7vMczFDzRKKg2aZRdYDI0x96sqhttRMJndyl1lY5chqcDZAUBQEhMI718LAi80atv3GuqeM\nLcpbZSaCC4zo2veVZpvfxe5/mXpNfwiSp8r9Cx7ilcvsPb/tef6272+svWGtjJTpYVprLv3DGsIC\nrXx430Xdf8DjO+GzR83KUgFhZmTNpLvBFg4f3GFm3hxzC1zxnBlDD5C9ET560MwzP/QyOLbJTHk8\n5xemy6f55Gwup5nFc+vrZkWpU4up2yLMXD99R0Ncmgnj2KFmxauOOO1QXQSlR2H3eybYG6ogfiSM\nv92U5fBKc6OXvQaUxdSnrhw47f8xKgXSroBhl0PSFNAuc5JY/VuoKYYLv2umrgiJMwupO+rMo70G\nynJMGUqPND5mmzuhI5ObfQyA+gqzqEzBfsjfB2XZ5tjRg83JI3Ec9BsHfS5s/U7qiuPmhrXtb5og\nH3alOSHVFEP/SSbkUxe472TZnNam/NVFUF1oRllFDzQrhbU1CZ/LZVZPC4xo2VjwEm5dQ1UpNR94\nAfADXtNan3GlSyk1C3ge8AeKtNYz29vnuYT7tUuvJSwgjL/O/+tZfb84N6+uyeLXnx5g2Y8vZmhC\nD/1z5G2DTX+GfYtNIAdGmLtcr3jOzMR5Okd94w1S/2ta61f90fzTt8flNP3wedvM4uR520yXkLNZ\nl01wrBnVY/E3oawsJtC0NjN7VhVAbck321uDzDq44/8D+k9o2Yp1NEDu15C1yrTWg6NN6zww0jxG\npZgTS2st37pyM7XDxhdNoLcnIAyiU0yQN1RD2TEoz2lZL+UHMUMgYQTEjzD1Or7DfFQ0m6c/OPab\nE0NU4/62v2VOOuNubZwvKdHcibzzHdjwf+Z4sanmYnlYP7O2QVhfs26BsphAriuH+nLzeWCEWUim\ntXUN7LXmZHzgI7NAe1UBOFtZVMYaaH528SPNwjW1pWYUVvFhc7Jz1AHKLDzfdzT0bTyRWwPN6yVH\nvnlEm4XnB80yf0v+gd8cp6HG/J3kbDLbDr/KNCi640TWyG3hrpTyAzKBS4Fc4GvgRq31/mbbRAIb\ngPla62NKqXitdUF7+z3bcC+oKWDOv+bw0LiHuOPCO7r8/eLcFVXVM+V/lvMf01J44soRPXvwihOw\n9S+me+OSn3XcLVFfaboyzrZrwOkwLdniQ1CUaT7Kc82JQLtMqGuX2X9wtOneCY03LenQBBNoQZFn\nd+zOKM8zJzwwwWS1ffMY3t+cIIKjz6y/ywVVJ03w+geb8G0eWs1VnjQhn7/PbH/qozzH/BzG3AgX\nP2rC/nROh1nAZsurZlK5mg4moGsucgAMmGa6dgJCTKAfXGbeXQVGmmkuIpPMzzokziw6HxAGJYdN\nWfP3mXcjVfnm/onoQebdSMwgM/VFTQmc2Gn+lspPHySgzBxJ0QNNQyFvm+mqsgaaE0/MEHPx/uRu\n804MzLuv+gpzjMn3wpibWr7TqSqE3C1mic3kqWe9lKY7w30q8JTWel7j148DaK1/02yb+4B+Wusn\nOlvAsw33Dw9+yJMbnuT9he+TFp3W5e8X7nHvW9vYcrSEDT+dTaB/N81BL3o3l8u0mk91h3WGvQ4q\nT5iPiuPmucAIE4yBERAYbk4mxzZC9gbTpVZTZLYLiYfhV5q5iFJmtLma2Bnqys0Jvr21EqqLzTUO\np90Ef2Ryy5NdXYUpT9Yq0+VUetQ0LJImm7DvP9F08exfApsWQd5W07U36jpz/Nwt5nvAvOu7+BGY\n9Vjnf27NuDPcr8W0yO9s/PpWYLLW+oFm25zqjhkJhAEvaK3fbGVfdwN3AyQnJ4/Pzs7ufI0aOVwO\n9hbtZXTcaBkp40EbDhVx02ubuTg1jpdvGU9QgAS86AZam3dNdeUmTLtrMRt3y/kaNr1kwj4kzqzG\n1n8SJE0y3T9dOSGepqfD/U/ABGAOEARsBK7QWme2td9z6XMXvcM/v87hscW7mZgSzev/MZFQm1cO\nvhKi+zjqTZeQGxuinQ33zvT65wFJzb7u3/hcc7nAF1rraq11EbAGGN3ZwgrvdN3EJF64YSzbsku5\n+bXNlNd0sByfEOcbq81jQ0E7E+5fA0OVUgOVUgHADcDS07ZZAkxXSlmVUsHAZKCDO0+EL7hqdD8W\n3TyOA8cruOHVTRRVtTJyQQjR4zoMd621A3gA+AIT2P/UWu9TSt2rlLq3cZsDwOfAbmALZrjk3u4r\ntuhNLhvZh9dun8CRoique3kjb248yoZDRRRU1HXfJGNCiHZ55U1MonfacqSE+97Z3qL1HhZoZUh8\nKAtH9eOmyckyskaIc+TWm5i6g4S7b9Jak19Rz+HCKg4VmI/deeXsyimjb0QgD84ZyrXj+2P188pp\njYTwOAl30ausP1TEs19ksDOnjIGxITw0dyhXjuqHn0WGswrRFRLuotfRWvPVgQKe+yKDjPxKAEIC\n/AgNtBIW6E+ozUpydDBzhsczKzWeiOBO3qQixHmks+EuA5NFj1FKcemIBOYMi+eLfSdJP1lJVb2D\nyjp746ODDYeLWLrrOH4WxYQBUcwdnsC0ITH0CQ8kKjgAi7T0hegUabmLXsXl0uzMLWP5gXyWHygg\n/WRl02t+FkVsaABxYTb6RwZz/cQkZqbGSeCL84p0ywifkFNSw67cMgor6ymqqqew0nzsO15BQWU9\ng+NCuGP6IL49LlFG4ojzgoS78GkNDhef7jnBq2uz2He8guiQAG6ZnMz1k5JJjOzavB21DU7e2ZyN\nzWrhlikDZM4i0atJuIvzgtaazUdKeG3tEZan5wNw0eBYvjuhP/NG9mm3NV9nd/LulmO8uPJw09j8\nS0ck8Nx3RxMR1LWLuVX1DplbR/QICXdx3skpqeH9bbm8vy2XvLJawmxWrhzdj1H9I4gLtREXZiM+\n3EZkUAAf7sjjjysOcqK8jskDo3l4Xhp788r59ScH6BcZxEs3j+OCxM4tM/fWpmyeXLKXb41N5Mkr\nRxAZHNDNNT07Wmt5V+IDJNzFecvl0mw6Usz723L5bM9Jau3OVrcbmxzJw5elMW1wTFPobcsu5YG/\nb6e4uoGnFo7kxklJ7Qbi53tP8IN3tpMaH8ahwiqigv351dUXcPmFfc8o09bsUj7adZyTFXU4nC4c\nLo3d6cLh1FyQGMH9lwwhLszmvh9EoxPltbyx4Sjvbclh7vAEfvPtCwmwyk1k3krCXQjA7nRRVFVP\nQYW5EFvQeGH2wsQIZqXFtRrcJdUNPPjeDtYeLOKaMf146qqRrbbGtxwp4Za/bGZkv3D+fucUsoqq\nePT93ew7XsH8kX341dUjqW5w8uH2XD7cmUdOSS3BAX4kRwfj72fB6qfwt1hAwfbsUgKsFu6+eBB3\nzRhEiBu6ePYdL+e1tUf4aNdxXFozMSWazUdKmDIompdvmSD3EXgpCXchzoHTpXlx5SFeWH6QqGB/\nfrFwJFeO6tt0MsjMr+TaRRuIDbPxwb3TiAox4e9wunh17RH+8FUmCqh3uFAKpg+J5dvjEpk3sg/B\nAWcG95Giap79Ip1P95wkNtTGg3OHcsPEJPw7OU1Dnd3J4cIqMvMryThZxfbsUrYcLSEkwI/rJybz\nvYtSSIoO5t878nj0/d0kRQfxxvcmkRQd7LafmegZEu5CuMH+4xX8dPFudueWM3tYPE9fcwEK+M6i\nDThdmsX3TaN/1JkBmVVYxcursxgUF8LVYxLpE9HG+qSn2X6slGc+TWfL0RKC/P2ICPInxOZHaKA/\noTY/Aq1+NDhd2J0uGhwu7E5NVb2DYyU1OF3mf9nfTzEkPoxrxvTjhknJZ1wc3pRVzD1vbcPfT/Ha\n7RMZk9SNa7wKt5NwF8JNnC7NX9cf4X+/zMSiIDo0gLJqO/+4Zyoj+oW7/Xhaa1akF7DuUBHV9Q6q\n6h1U1TupqrNTZ3fhb7Vg87MQYDUfQf5+DIoLIa1PGGkJYaTEhnTY4j9cWMX3/vo1BZV1PDQ3lTnD\n4hkSH3pGN1VVvYPlB/L5dM8JSmvsPDY/jfEDot1e57Y4XZqPdx/naFENN05KIj68/ZNkg8OF1aJ8\n+sY2CXch3CynpIaf/3svm7OK+ev3JjJtcKyni3ROiqvque+d7Ww+UgJAn/BAZgyNZfrQWJRSfLr7\nBCszCqh3uIgPs2FRivzKOm6fmsIj89LOuC6QW1rDq2uyWLw9j+jQAAbHhTIkPpTBcSEMiQ9lVP/I\nTncznTrBPftFRtNdyjarhZsnD+DemYNahHyDw8WK9Hze+zqH1ZmFAIQGWAkNtBJqsxIWaGX8gCgu\nHdGH8QOiznmyOpdLU1hVT1RwgEcuTEu4C9ENtNbU2V0+tSB4bmkN6w4WsfZgEesOFVFea5ZLjAuz\ncfkFfbhiVD8mDIiixu7kd5+n8+bGbPpHBfGbb1/IjKFxHC6sYtGqw/x7Rx5KwYIL+uLUmsMFVWQV\nVdPgcAEQH2bjxknJ3DQ5mYR2WuBfHy3ht5+lszW7lJSYYH5yWRoXJEbw0spDLN6Rh9WiuGXKAK4Y\n1Zcv9p7kg+25FFU10Cc8kKvG9CPQajHvdOrNnEVFVQ3sPFZGg9NFTEgAs4fFc+mIBCKC/DlZUUd+\nRR0nyusoqKgnLszGNWMTGd0/4ox3MbUNThbvyOUv646QVVgNQHRIAPFhNuLDA0kIs3HfJUMYGBvS\nTb8pQ8JdCNFlTpdmT145DqeLscmtt3K3HCnhpx/sJquomjFJkezKLcNmtXDDxGTuvngQ/ZrdIex0\naXJLa9h3vIJ/bs1hVUYhVoti3sg+3Dp1AH3CA8nMr+RgQRUH8ytJP2k+4sPMReXrJrS8qHy0qJo/\nrTzEhzvycLo0VotizvB4bpiYzMWpcW22yivr7KzOLOTLffmsTC+gst7R4vXgAD8SwgPJK6ulweFi\nUFwI3x6byNVjErH5W3hrYzZvb8qmtMbOhYkRXD2mHzUNTvIr6sivqKew0jy+8f2JDOvj/q665iTc\nhRDdps7u5IXlB/l0zwmuuLAv358+kNjQjsfoHy2q5u1N2fxzaw4VdS0DNjEyiKEJoVw0OJZbpgxo\n993R0aJqvj5awqy0+C7fG9DgcLE1uwSnS9MnPJCEiEDCbFaUUpTX2vlszwkW78hjS2N3lb+fwuHS\nzB2ewJ3TBzJpYLRHbwaTcBdC9Fq1DU4+23sCh1MzNMH0zYcF9q5x9zklNSzddZyKWjs3TEru9u6W\nzpJwF0IIH9TZcJd7kIUQwgdJuAshhA+ScBdCCB8k4S6EED5Iwl0IIXyQhLsQQvggCXchhPBBEu5C\nCOGDPHYTk1KqEMg+y2+PBYrcWBxPk/r0Xr5UF/Ct+vhSXaDz9RmgtY7raCOPhfu5UEpt7cwdWt5C\n6tN7+VJdwLfq40t1AffXR7plhBDCB0m4CyGED/LWcH/F0wVwM6lP7+VLdQHfqo8v1QXcXB+v7HMX\nQgjRPm9tuQshhGiH14W7Umq+UipDKXVIKfVTT5enq5RSryulCpRSe5s9F62UWqaUOtj4GOXJMnaW\nUipJKbVSKbVfKbVPKfVg4/PeWp9ApdQWpdSuxvr8svF5r6wPgFLKTym1Qyn1cePX3lyXo0qpPUqp\nnUqprY3PeWV9lFKRSqn3lVLpSqkDSqmp7q6LV4W7UsoPeBFYAIwAblRKjfBsqbrsDWD+ac/9FFiu\ntR4KLG/82hs4gJ9orUcAU4D7G38f3lqfemC21no0MAaYr5SagvfWB+BB4ECzr725LgCXaK3HNBsy\n6K31eQH4XGs9DBiN+R25ty5aa6/5AKYCXzT7+nHgcU+X6yzqkQLsbfZ1BtC38fO+QIany3iW9VoC\nXOoL9QGCge3AZG+tD9C/MSRmAx83PueVdWks71Eg9rTnvK4+QARwhMZrnt1VF69quQOJQE6zr3Mb\nn/N2CVrrE42fnwQSPFmYs6GUSgHGApvx4vo0dmPsBAqAZVprb67P88CjgKvZc95aFwANfKWU2qaU\nurvxOW+sz0CgEPhrY5fZa0qpENxcF28Ld5+nzWnbq4YwKaVCgQ+Ah7TWFc1f87b6aK2dWusxmFbv\nJKXUBae97hX1UUpdCRRorbe1tY231KWZ6Y2/mwWYLsCLm7/oRfWxAuOARVrrsUA1p3XBuKMu3hbu\neUBSs6/7Nz7n7fKVUn0BGh8LPFyeTlNK+WOC/R2t9eLGp722PqdorcuAlZjrI95Yn4uAq5RSR4H3\ngNlKqbfxzroAoLXOa3wsAD4EJuGd9ckFchvfFQK8jwl7t9bF28L9a2CoUmqgUioAuAFY6uEyucNS\n4PbGz2/H9F33ekopBfwFOKC1/n2zl7y1PnFKqcjGz4Mw1w/S8cL6aK0f11r311qnYP5PVmitb8EL\n6wKglApRSoWd+hy4DNiLF9ZHa30SyFFKpTU+NQfYj7vr4umLC2dxMeJyIBM4DPzc0+U5i/K/C5wA\n7Jgz+B1ADObC10HgKyDa0+XsZF2mY9467gZ2Nn5c7sX1GQXsaKzPXuDJxue9sj7N6jWLby6oemVd\ngEHArsaPfaf+9724PmOArY1/a/8GotxdF7lDVQghfJC3dcsIIYToBAl3IYTwQRLuQgjhgyTchRDC\nB0m4CyGED5JwF0IIHyThLoQQPkjCXQghfND/AxIebeIYuVSLAAAAAElFTkSuQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fc411f54fd0>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

## 对测试集分类

当得到一组满意的模型设计和参数后，我们使用全部训练数据集（含验证集）重新训练模型，并对测试集分类。

```{.python .input  n=8}
filename = "./mydog_resnet50-v1_train.params"
net.export(filename)
```

```{.python .input  n=11}
import numpy as np

net = get_net(ctx)
net.hybridize()
train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, 
      ctx, lr_period, lr_decay)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Train loss: 2.975789, Time 00:15:17, lr 0.001\nEpoch 1. Train loss: 1.451438, Time 00:15:18, lr 0.001\nEpoch 2. Train loss: 1.215039, Time 00:15:28, lr 0.001\nEpoch 3. Train loss: 1.093664, Time 00:15:26, lr 0.001\nEpoch 4. Train loss: 0.985698, Time 00:15:26, lr 0.001\nEpoch 5. Train loss: 0.934325, Time 00:15:27, lr 0.001\nEpoch 6. Train loss: 0.907355, Time 00:15:29, lr 0.001\nEpoch 7. Train loss: 0.840103, Time 00:15:24, lr 0.001\nEpoch 8. Train loss: 0.806787, Time 00:15:31, lr 0.001\nEpoch 9. Train loss: 0.806220, Time 00:15:29, lr 0.001\nEpoch 10. Train loss: 0.757296, Time 00:15:16, lr 0.001\nEpoch 11. Train loss: 0.744249, Time 00:15:11, lr 0.001\nEpoch 12. Train loss: 0.711346, Time 00:15:26, lr 0.001\nEpoch 13. Train loss: 0.664984, Time 00:15:27, lr 0.001\nEpoch 14. Train loss: 0.678449, Time 00:15:23, lr 0.001\nEpoch 15. Train loss: 0.656187, Time 00:15:29, lr 0.001\nEpoch 16. Train loss: 0.636772, Time 00:15:28, lr 0.001\nEpoch 17. Train loss: 0.609805, Time 00:15:26, lr 0.001\nEpoch 18. Train loss: 0.602244, Time 00:15:21, lr 0.001\nEpoch 19. Train loss: 0.599717, Time 00:15:23, lr 0.001\nEpoch 20. Train loss: 0.529028, Time 00:15:22, lr 0.0001\nEpoch 21. Train loss: 0.499422, Time 00:15:24, lr 0.0001\nEpoch 22. Train loss: 0.481691, Time 00:15:23, lr 0.0001\nEpoch 23. Train loss: 0.473554, Time 00:15:22, lr 0.0001\nEpoch 24. Train loss: 0.455049, Time 00:15:20, lr 0.0001\nEpoch 25. Train loss: 0.455707, Time 00:15:16, lr 0.0001\nEpoch 26. Train loss: 0.452966, Time 00:15:15, lr 0.0001\nEpoch 27. Train loss: 0.463867, Time 00:15:12, lr 0.0001\nEpoch 28. Train loss: 0.458508, Time 00:15:15, lr 0.0001\nEpoch 29. Train loss: 0.449892, Time 00:15:08, lr 0.0001\nEpoch 30. Train loss: 0.446990, Time 00:15:07, lr 0.0001\nEpoch 31. Train loss: 0.423945, Time 00:15:06, lr 0.0001\nEpoch 32. Train loss: 0.439685, Time 00:15:06, lr 0.0001\nEpoch 33. Train loss: 0.418496, Time 00:15:07, lr 0.0001\nEpoch 34. Train loss: 0.437367, Time 00:15:04, lr 0.0001\nEpoch 35. Train loss: 0.417096, Time 00:15:05, lr 0.0001\nEpoch 36. Train loss: 0.422807, Time 00:15:07, lr 0.0001\nEpoch 37. Train loss: 0.410981, Time 00:15:05, lr 0.0001\nEpoch 38. Train loss: 0.421435, Time 00:15:06, lr 0.0001\nEpoch 39. Train loss: 0.410702, Time 00:15:07, lr 0.0001\nEpoch 40. Train loss: 0.421057, Time 00:15:06, lr 1e-05\nEpoch 41. Train loss: 0.408965, Time 00:15:05, lr 1e-05\nEpoch 42. Train loss: 0.407438, Time 00:15:07, lr 1e-05\nEpoch 43. Train loss: 0.400689, Time 00:15:02, lr 1e-05\nEpoch 44. Train loss: 0.405391, Time 00:15:08, lr 1e-05\nEpoch 45. Train loss: 0.415739, Time 00:15:06, lr 1e-05\nEpoch 46. Train loss: 0.385631, Time 00:15:05, lr 1e-05\nEpoch 47. Train loss: 0.405506, Time 00:15:05, lr 1e-05\nEpoch 48. Train loss: 0.402261, Time 00:15:05, lr 1e-05\nEpoch 49. Train loss: 0.410830, Time 00:15:04, lr 1e-05\nEpoch 50. Train loss: 0.399737, Time 00:15:04, lr 1e-05\nEpoch 51. Train loss: 0.404163, Time 00:15:07, lr 1e-05\nEpoch 52. Train loss: 0.384466, Time 00:15:06, lr 1e-05\nEpoch 53. Train loss: 0.405358, Time 00:15:05, lr 1e-05\nEpoch 54. Train loss: 0.405773, Time 00:15:06, lr 1e-05\nEpoch 55. Train loss: 0.413891, Time 00:15:06, lr 1e-05\nEpoch 56. Train loss: 0.406771, Time 00:15:06, lr 1e-05\nEpoch 57. Train loss: 0.384557, Time 00:15:10, lr 1e-05\nEpoch 58. Train loss: 0.397814, Time 00:15:20, lr 1e-05\nEpoch 59. Train loss: 0.391230, Time 00:15:21, lr 1e-05\n"
 }
]
```

```{.python .input  n=12}
filename = "./mydog_resnet50-v1_train_valid.params"
net.export(filename)
```

```{.python .input  n=13}
import os
outputs = []
for data, label in test_data:
    output = nd.softmax(net(data.as_in_context(ctx)))
    outputs.extend(output.asnumpy())
ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('resnet50-v1.csv', 'w') as f:
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
