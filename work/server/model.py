"""Model definition."""   
from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

class Flatten(nn.Module):
    """
    定义一个Flatten类,继承自nn.Module,用于展平输入张量。

    此模块将接收一个任意形状的输入张量，并将其除了第一维（批次大小）以外的所有维度展平成一维。

    方法:
        forward(x): 实现张量的展平操作。
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        执行展平操作。

        参数:
        x (Tensor): 输入的多维张量。

        返回:
        Tensor: 展平后的二维张量，形状为 (batch_size, num_features)。
        """
        return x.view(x.size(0), -1)

class Model(nn.Module):
    """
    Model类是一个为视频理解任务设计的深度学习模型。
    其使用预训练的基础模型(如ResNet152)来提取特征,并通过Temporal Segment Networks(TSN)进行视频中的动作识别。
    适用于处理分割成多个片段的视频数据，并能够根据不同的输入表示(mv,residual)进行调整。

    参数:
        num_class (int): 分类任务的类别数。
        num_segments (int): 视频被分割成的片段数。
        representation (str): 输入数据的表示形式，例如 'mv' 或 'residual'。
        base_model (str): 用作特征提取的预训练模型名称，默认为 'resnet152'。

    属性:
        _representation (str): 表示输入数据的形式。
        num_segments (int): 视频片段的数量。
        _input_size (int): 模型输入图像的尺寸。
        base_model (nn.Module): 预训练的基础模型。
        data_bn (nn.Module): 应用于输入数据的批量归一化层。

    方法:
        _prepare_tsn(num_class): 准备时间同步网络(TSN)的相关设置。
        _prepare_base_model(base_model): 根据指定的基础模型名称加载预训练模型。
        forward(input): 定义模型的前向传播过程。
        get_augmentation(): 返回用于数据增强的变换组合。
    """

    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152'):
        """
        初始化模型的构造函数。

        设置模型的基本参数,并准备基础模型和TSN。

        参数:
            num_class (int): 分类任务的类别数。
            num_segments (int): 视频被分割成的片段数。
            representation (str): 输入数据的表示形式。
            base_model (str): 基础模型的名称。
        """
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):
        """
        准备 temparol segment networks (TSN,一个用于视频中动作识别的卷积网络体系结构)的相关设置。

        根据类别数调整基础模型的全连接层，并根据输入类型调整卷积层。
        如果输入的表示形式为运动矢量 (mv)，模型会调整其第一层卷积网络来处理两个通道的输入，并应用相应的批量归一化层。
        如果输入的表示形式为残差，模型会处理帧之间的差异，而不是单独的帧

        参数:
            num_class (int): 分类任务的类别数。
        """
        feature_dim = getattr(self.base_model, 'fc').in_features #获取基础模型（如ResNet152）中全连接层 (fc) 的输入特征数量
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class)) #将基础模型的全连接层替换为一个新的全连接层，输入维度不变，输出维度为分类任务的类别数num_class。

        if self._representation == 'mv': 
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2, 64, 
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False)) #如果输入的表示形式为'mv',修改基础模型的第一层卷积层，使其接受2个通道的输入（通常是运动矢量的两个分量）。
            self.data_bn = nn.BatchNorm2d(2) #添加一个2通道的批量归一化层
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3) #如果输入的表示形式为残差，代码会添加一个3通道的批量归一化层，


    def _prepare_base_model(self, base_model):
        """
        根据指定的基础模型名称加载预训练模型。

        如果基础模型是ResNet系列,则加载相应的预训练模型,并设置224*224尺寸的图像作为模型的输入。
        否则，抛出一个错误。

        参数:
            base_model (str): 基础模型的名称。
        """
        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True) #从torchvision.models中加载名为 base_model的模型

            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        """
        定义模型的前向传播过程。

        调整输入的形状，应用批量归一化，然后通过基础模型进行处理。

        参数:
            input (Tensor): 输入数据的张量。

        返回:
            Tensor: 基础模型(base_model)的输出结果,base_out即为基础模型提取的特征,用于后续的动作识别。
        """
        input = input.view((-1, ) + input.size()[-3:]) #重新塑形输入张量。保持最后三个维度不变，而第一个维度（-1）将由PyTorch自动计算，以确保总的元素数量保持不变。
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)#对输入数据进行批量归一化

        base_out = self.base_model(input)#将处理过的输入数据传递给基础模型（self.base_model，得到模型输出
        return base_out

    @property
    def crop_size(self):
        """
        获取模型预期的输入图像裁剪尺寸。

        返回:
            int: 图像裁剪尺寸，与模型训练时使用的尺寸一致。
        """
        return self._input_size

    @property
    def scale_size(self):
        """
        获取模型预期的输入图像缩放尺寸。

        缩放尺寸是基于裁剪尺寸计算得出的，用于在裁剪前调整图像大小。

        返回:
            int: 图像缩放尺寸，通常比裁剪尺寸稍大，以便于裁剪。
        """
        return self._input_size * 256 // 224

    def get_augmentation(self):
        """
        根据模型的输入表示获取数据增强的变换组合。这些变换可以在训练时应用于输入数据，以增加数据的多样性和模型的鲁棒性。

        如果输入表示为 'mv' 或 'residual'，则使用一组较小的尺度列表；否则，使用一个包含更多尺度的列表。
        这些尺度用于多尺度裁剪，以及是否应用随机水平翻转。

        返回:
            Compose: 一个包含多个数据增强变换的组合对象。
        """
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales), #多尺度裁剪操作，根据尺度（scales）和输入尺寸（self._input_size）随机裁剪图像。
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))]) #随机水平翻转操作，根据输入表示是否为'mv'来决定是否应用翻转。
