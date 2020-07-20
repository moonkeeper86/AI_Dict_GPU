
import torch.nn as nn
import torch.nn.functional as F
import torch

# AlexNet
class AlexNet(nn.Module):   
    input_resize = (227, 227, 3)
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# LeNet_5
class LeNet_5(nn.Module):
    input_resize = (28, 28)
    def __init__(self, num_classes=10):
        # nn.Module子类必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(LeNet_5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    def forward(self, x):
        x = self.features(x.cuda())
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

#202006180847 NIN_Block
class NIN_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(NIN_Block, self).__init__()
        self.nin_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1,
                      stride=1, padding=0),  # 1x1卷积,整合多个feature map的特征
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1,
                      stride=1, padding=0),  # 1x1卷积,整合多个feature map的特征
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.nin_block(x)
        return x

#NetworkinNetwork
class NetworkinNetwork(nn.Module):
    input_resize = (224, 224)
    def __init__(self, num_classes=10):
        super(NetworkinNetwork, self).__init__()
        self.classifier = nn.Sequential(
            NIN_Block(1, 96, 11, 4, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NIN_Block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NIN_Block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NIN_Block(384, num_classes, kernel_size=3,
                            stride=1, padding=1)
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6, stride=1)
        )
    def forward(self, x):
        feature = self.classifier(x)
        output = self.gap(feature)
        output = output.view(x.shape[0], -1)  # [batch,10,1,1]-->[batch,10]
        return output

# Inception_Out_Block
class Inception_Out_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, input_chs):
        super(Inception_Out_Block, self).__init__()
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(input_chs, 128, kernel_size=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128 * 4 * 4, 1024),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 2),  # 后一个参数是分类数量，针对Coal设定为2
            # Softmax(),
            # ReLU(),
        )
    def forward(self, x):
        x = self.pool1(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# InceptionV1_Module_Block
class InceptionV1_Module_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, input_chs, model_size):
        super(InceptionV1_Module_Block, self).__init__()
        con1_chs, con31_chs, con3_chs, con51_chs, con5_chs, pool11_chs = model_size
        self.InceptionV1_Module_Block_branch_conv1 = nn.Sequential(
            nn.Conv2d(input_chs, con1_chs, kernel_size=1),
            nn.ReLU(),
        )
        self.InceptionV1_Module_Block_branch_conv3 = nn.Sequential(
            nn.Conv2d(input_chs, con31_chs, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(con31_chs, con3_chs, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.InceptionV1_Module_Block_branch_conv5 = nn.Sequential(
            nn.Conv2d(input_chs, con51_chs, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(con51_chs, con5_chs, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.InceptionV1_Module_Block_branch_pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(input_chs, pool11_chs, kernel_size=1),
            nn.ReLU(),
        )
    def forward(self, x):
        out1 = self.InceptionV1_Module_Block_branch_conv1(x)
        out2 = self.InceptionV1_Module_Block_branch_conv3(x)
        out3 = self.InceptionV1_Module_Block_branch_conv5(x)
        out4 = self.InceptionV1_Module_Block_branch_pool1(x)
        result = torch.cat([out1, out2, out3, out4], dim=1)
        return result

# 202006171800 InceptionV1
class InceptionV1(nn.Module):
    input_resize = (224, 224, 3)
    def __init__(self, num_classes=10):
        super(InceptionV1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception3a = InceptionV1_Module_Block(
            192, (64, 96, 128, 16, 32, 32))
        self.inception3b = InceptionV1_Module_Block(
            256, (128, 128, 192, 32, 96, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionV1_Module_Block(
            480, (192, 96, 208, 16, 48, 64))
        if self.training == True:
            self.out1 = Inception_Out_Block(512)
        self.inception4b = InceptionV1_Module_Block(
            512, (160, 112, 224, 24, 64, 64))
        self.inception4c = InceptionV1_Module_Block(
            512, (128, 128, 256, 24, 64, 64))
        self.inception4d = InceptionV1_Module_Block(
            512, (112, 144, 288, 32, 64, 64))
        if self.training == True:
            self.out2 = Inception_Out_Block(528)
        self.inception4e = InceptionV1_Module_Block(
            528, (256, 160, 320, 32, 128, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionV1_Module_Block(
            832, (256, 160, 320, 32, 128, 128))
        self.inception5b = InceptionV1_Module_Block(
            832, (384, 192, 384, 48, 128, 128))
        self.pool3 = nn.AvgPool2d(kernel_size=7, stride=1, )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes),  # 后一个参数是分类数量，针对Coal设定为2
            # Softmax(),
            # ReLU(),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool1(x)
        x = self.inception4a(x)
        if self.training == True:
            output1 = self.out1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training == True:
            output2 = self.out2(x)
        x = self.inception4e(x)
        x = self.pool2(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        if self.training == True:
            return output1, output2, output
        else:
            return output

# ConvBNReLU
class ConvBNReLU(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.ConvBNReLU = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        x = self.ConvBNReLU(x)
        return x

# ConvBNReLUFactorization
class ConvBNReLUFactorization(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings):
        super(ConvBNReLUFactorization, self).__init__()
        self.ConvBNReLUF = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_sizes, stride=1, padding=paddings),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_sizes, stride=1, padding=paddings),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        x = self.ConvBNReLUF(x)
        return x

# InceptionV2_ModuleA_Block
class InceptionV2_ModuleA_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2_ModuleA_Block, self).__init__()
        self.InceptionV2_ModuleA_Block_branch1 = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.InceptionV2_ModuleA_Block_branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce,
                             out_channels=out_channels2, kernel_size=3, padding=1),
        )
        self.InceptionV2_ModuleA_Block_branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels3reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce,
                             out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3,
                             out_channels=out_channels3, kernel_size=3, padding=1),
        )
        self.InceptionV2_ModuleA_Block_branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels4, kernel_size=1),
        )
    def forward(self, x):
        out1 = self.InceptionV2_ModuleA_Block_branch1(x)
        out2 = self.InceptionV2_ModuleA_Block_branch2(x)
        out3 = self.InceptionV2_ModuleA_Block_branch3(x)
        out4 = self.InceptionV2_ModuleA_Block_branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

# InceptionV2_ModuleB_Block
class InceptionV2_ModuleB_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2_ModuleB_Block, self).__init__()
        self.InceptionV2_ModuleB_Block_branch1 = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.InceptionV2_ModuleB_Block_branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
                                          out_channels=out_channels2reduce, kernel_sizes=[1, 3], paddings=[0, 1]),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
                                          out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0]),
        )
        self.InceptionV2_ModuleB_Block_branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels3reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
                                          out_channels=out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
                                          out_channels=out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
                                          out_channels=out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
                                          out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1]),
        )
        self.InceptionV2_ModuleB_Block_branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels4, kernel_size=1),
        )
    def forward(self, x):
        out1 = self.InceptionV2_ModuleB_Block_branch1(x)
        out2 = self.InceptionV2_ModuleB_Block_branch2(x)
        out3 = self.InceptionV2_ModuleB_Block_branch3(x)
        out4 = self.InceptionV2_ModuleB_Block_branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

# InceptionV2_ModuleC_Block
class InceptionV2_ModuleC_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2_ModuleC_Block, self).__init__()
        self.InceptionV2_ModuleC_Block_branch1 = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.InceptionV2_ModuleC_Block_branch2_conv1 = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.InceptionV2_ModuleC_Block_branch2_conv2a = ConvBNReLUFactorization(
            in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1, 3], paddings=[0, 1])
        self.InceptionV2_ModuleC_Block_branch2_conv2b = ConvBNReLUFactorization(
            in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0])
        self.InceptionV2_ModuleC_Block_branch3_conv1 = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.InceptionV2_ModuleC_Block_branch3_conv2 = ConvBNReLU(
            in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.InceptionV2_ModuleC_Block_branch3_conv3a = ConvBNReLUFactorization(
            in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0])
        self.InceptionV2_ModuleC_Block_branch3_conv3b = ConvBNReLUFactorization(
            in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1])
        self.InceptionV2_ModuleC_Block_branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels4, kernel_size=1),
        )
    def forward(self, x):
        out1 = self.InceptionV2_ModuleC_Block_branch1(x)
        x2 = self.InceptionV2_ModuleC_Block_branch2_conv1(x)
        out2 = torch.cat([self.InceptionV2_ModuleC_Block_branch2_conv2a(
            x2), self.InceptionV2_ModuleC_Block_branch2_conv2b(x2)], dim=1)
        x3 = self.InceptionV2_ModuleC_Block_branch3_conv2(
            self.InceptionV2_ModuleC_Block_branch3_conv1(x))
        out3 = torch.cat([self.InceptionV2_ModuleC_Block_branch3_conv3a(
            x3), self.InceptionV2_ModuleC_Block_branch3_conv3b(x3)], dim=1)
        out4 = self.InceptionV2_ModuleC_Block_branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

# InceptionV2_ModuleD_Block
class InceptionV2_ModuleD_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV2_ModuleD_Block, self).__init__()
        self.InceptionV2_ModuleD_Block_branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels1, kernel_size=3, stride=2, padding=1)
        )
        self.InceptionV2_ModuleD_Block_branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce,
                             out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2,
                             kernel_size=3, stride=2, padding=1),
        )
        self.InceptionV2_ModuleD_Block_branch3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        out1 = self.InceptionV2_ModuleD_Block_branch1(x)
        out2 = self.InceptionV2_ModuleD_Block_branch2(x)
        out3 = self.InceptionV2_ModuleD_Block_branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

# InceptionV2_ModuleE_Block
class InceptionV2_ModuleE_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV2_ModuleE_Block, self).__init__()
        self.InceptionV2_ModuleE_Block_branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce,
                             out_channels=out_channels1, kernel_size=3, stride=2),
        )
        self.InceptionV2_ModuleE_Block_branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
                                          out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings=[0, 3]),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
                                          out_channels=out_channels2reduce, kernel_sizes=[7, 1], paddings=[3, 0]),
            ConvBNReLU(in_channels=out_channels2reduce,
                             out_channels=out_channels2, kernel_size=3, stride=2),
        )
        self.InceptionV2_ModuleE_Block_branch3 = nn.MaxPool2d(
            kernel_size=3, stride=2)
    def forward(self, x):
        out1 = self.InceptionV2_ModuleE_Block_branch1(x)
        out2 = self.InceptionV2_ModuleE_Block_branch2(x)
        out3 = self.InceptionV2_ModuleE_Block_branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

# 202006171800 InceptionV2
class InceptionV2(nn.Module):
    input_resize = (224, 224, 3)
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV2, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=64,
                             kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=192,
                             kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block3 = nn.Sequential(
            InceptionV2_ModuleA_Block(in_channels=192, out_channels1=64, out_channels2reduce=64,
                                            out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV2_ModuleA_Block(in_channels=256, out_channels1=64, out_channels2reduce=64,
                                            out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=64),
            InceptionV3_Module_Block(in_channels=320, out_channels1reduce=128,
                                           out_channels1=160, out_channels2reduce=64, out_channels2=96),
        )
        self.block4 = nn.Sequential(
            InceptionV2_ModuleB_Block(in_channels=576, out_channels1=224, out_channels2reduce=64,
                                            out_channels2=96, out_channels3reduce=96, out_channels3=128, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=576, out_channels1=192, out_channels2reduce=96,
                                            out_channels2=128, out_channels3reduce=96, out_channels3=128, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=576, out_channels1=160, out_channels2reduce=128,
                                            out_channels2=160, out_channels3reduce=128, out_channels3=128, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=576, out_channels1=96, out_channels2reduce=128,
                                            out_channels2=192, out_channels3reduce=160, out_channels3=160, out_channels4=128),
            InceptionV3_Module_Block(in_channels=576, out_channels1reduce=128,
                                           out_channels1=192, out_channels2reduce=192, out_channels2=256),
        )
        self.block5 = nn.Sequential(
            InceptionV2_ModuleC_Block(in_channels=1024, out_channels1=352, out_channels2reduce=192,
                                            out_channels2=160, out_channels3reduce=160, out_channels3=112, out_channels4=128),
            InceptionV2_ModuleC_Block(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160,
                                            out_channels3reduce=192, out_channels3=112, out_channels4=128)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

# InceptionV3_Module_Block
class InceptionV3_Module_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3_Module_Block, self).__init__()
        self.InceptionV3_Module_Block_branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce,
                             out_channels=out_channels1, kernel_size=3, stride=2, padding=1)
        )
        self.InceptionV3_Module_Block_branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce,
                             out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2,
                             kernel_size=3, stride=2, padding=1),
        )
        self.InceptionV3_Module_Block_branch3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        out1 = self.InceptionV3_Module_Block_branch1(x)
        out2 = self.InceptionV3_Module_Block_branch2(x)
        out3 = self.InceptionV3_Module_Block_branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

# Inception_Aux_Block
class Inception_Aux_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception_Aux_Block, self).__init__()
        self.AUX_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.AUX_conv1 = ConvBNReLU(
            in_channels=in_channels, out_channels=128, kernel_size=1)
        self.AUX_conv2 = nn.Conv2d(
            in_channels=128, out_channels=768, kernel_size=5, stride=1)
        self.AUX_dropout = nn.Dropout(p=0.7)
        self.AUX_linear1 = nn.Linear(
            in_features=768, out_features=out_channels)
    def forward(self, x):
        x = self.AUX_conv1(self.AUX_avgpool(x))
        x = self.AUX_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.AUX_linear1(self.AUX_dropout(x))
        return out

# InceptionV3
class InceptionV3(nn.Module):
    input_resize = (224, 224, 3)
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV3, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=32,
                             kernel_size=3, stride=2),
            ConvBNReLU(in_channels=32, out_channels=32,
                             kernel_size=3, stride=1),
            ConvBNReLU(in_channels=32, out_channels=64,
                             kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=80,
                             kernel_size=3, stride=1),
            ConvBNReLU(in_channels=80, out_channels=192,
                             kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block3 = nn.Sequential(
            InceptionV2_ModuleA_Block(in_channels=192, out_channels1=64, out_channels2reduce=48,
                                      out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV2_ModuleA_Block(in_channels=256, out_channels1=64, out_channels2reduce=48,
                                      out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64),
            InceptionV2_ModuleA_Block(in_channels=288, out_channels1=64, out_channels2reduce=48,
                                      out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64)
        )
        self.block4 = nn.Sequential(
            InceptionV2_ModuleD_Block(in_channels=288,out_channels1=384,
                                      out_channels2reduce=64, out_channels2=96),
            InceptionV2_ModuleB_Block(in_channels=768, out_channels1=192, out_channels2reduce=128,
                                      out_channels2=192, out_channels3reduce=128, out_channels3=192, out_channels4=192),
            InceptionV2_ModuleB_Block(in_channels=768, out_channels1=192, out_channels2reduce=160,
                                      out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192),
            InceptionV2_ModuleB_Block(in_channels=768, out_channels1=192, out_channels2reduce=160,
                                      out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192),
            InceptionV2_ModuleB_Block(in_channels=768, out_channels1=192, out_channels2reduce=192,
                                      out_channels2=192, out_channels3reduce=192, out_channels3=192, out_channels4=192),
        )
        if self.stage == 'train':
            self.aux_logits = Inception_Aux_Block(
                in_channels=768, out_channels=num_classes)
        self.block5 = nn.Sequential(
            InceptionV2_ModuleE_Block(in_channels=768, out_channels1reduce=192,
                                      out_channels1=320, out_channels2reduce=192, out_channels2=192),
            InceptionV2_ModuleC_Block(in_channels=1280, out_channels1=320, out_channels2reduce=384,
                                      out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192),
            InceptionV2_ModuleC_Block(in_channels=2048, out_channels1=320, out_channels2reduce=384,
                                      out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux = x = self.block4(x)
        x = self.block5(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)

        if self.stage == 'train':
            aux = self.aux_logits(aux)
            return aux, out
        else:
            return out

# Inception_Mix3A_Block
class Inception_Mix3A_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self):
        super(Inception_Mix3A_Block, self).__init__()
        self.Inception_Mix3A_Block_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2)
        self.Inception_Mix3A_Block_conv = ConvBNReLU(
            64, 96, kernel_size=3, stride=2)
    def forward(self, x):
        x0 = self.Inception_Mix3A_Block_maxpool(x)
        x1 = self.Inception_Mix3A_Block_conv(x)
        out = torch.cat((x0, x1), 1)
        return out

# Inception_Mix4A_Block
class Inception_Mix4A_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self):
        super(Inception_Mix4A_Block, self).__init__()
        self.Inception_Mix4A_Block_branch0 = nn.Sequential(
            ConvBNReLU(160, 64, kernel_size=1, stride=1),
            ConvBNReLU(64, 96, kernel_size=3, stride=1)
        )
        self.Inception_Mix4A_Block_branch1 = nn.Sequential(
            ConvBNReLU(160, 64, kernel_size=1, stride=1),
            ConvBNReLU(64, 64, kernel_size=(
                1, 7), stride=1, padding=(0, 3)),
            ConvBNReLU(64, 64, kernel_size=(
                7, 1), stride=1, padding=(3, 0)),
            ConvBNReLU(64, 96, kernel_size=(3, 3), stride=1)
        )
    def forward(self, x):
        x0 = self.Inception_Mix4A_Block_branch0(x)
        x1 = self.Inception_Mix4A_Block_branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

# Inception_Mix5A_Block
class Inception_Mix5A_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self):
        super(Inception_Mix5A_Block, self).__init__()
        self.Inception_Mix5A_Block_conv = ConvBNReLU(
            192, 192, kernel_size=3, stride=2)
        self.Inception_Mix5A_Block_maxpool = nn.MaxPool2d(3, stride=2)
    def forward(self, x):
        x0 = self.Inception_Mix5A_Block_conv(x)
        x1 = self.Inception_Mix5A_Block_maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out

# InceptionV4_Stem_Block
class InceptionV4_Stem_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self):
        super(InceptionV4_Stem_Block, self).__init__()
        self.InceptionV4_Stem = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=32,
                             kernel_size=3, stride=2),
            ConvBNReLU(in_channels=32, out_channels=32,
                             kernel_size=3, stride=1),
            ConvBNReLU(in_channels=32, out_channels=64,
                             kernel_size=3, stride=1, padding=1),
            Inception_Mix3A_Block(),
            Inception_Mix4A_Block(),
            Inception_Mix5A_Block()
        )
    def forward(self, x):
        x = self.InceptionV4_Stem(x)
        return x

# InceptionV4_ReductionA_Block
class InceptionV4_ReductionA_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1, out_channels2reduceA, out_channels2reduceB, out_channels2):
        super(InceptionV4_ReductionA_Block, self).__init__()
        self.InceptionV4_ReductionA_Block_branch1 = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels1, kernel_size=1, stride=2)  # 20200715 加入stride=2保证3分支输出等维度
        self.InceptionV4_ReductionA_Block_branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels2reduceA, kernel_size=1, stride=2),   # 20200715 加入stride=2保证3分支输出等维度
            ConvBNReLUFactorization(in_channels=out_channels2reduceA,
                                          out_channels=out_channels2reduceB, kernel_sizes=[1, 3], paddings=[0, 1]),
            ConvBNReLUFactorization(in_channels=out_channels2reduceB,
                                          out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0]),
        )
        self.InceptionV4_ReductionA_Block_branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    def forward(self, x):
        out1 = self.InceptionV4_ReductionA_Block_branch1(x)
        out2 = self.InceptionV4_ReductionA_Block_branch2(x)
        out3 = self.InceptionV4_ReductionA_Block_branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

# InceptionV4_ReductionB_Block
class InceptionV4_ReductionB_Block(nn.Module):
    input_resize = (28, 28)
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV4_ReductionB_Block, self).__init__()
        self.InceptionV4_ReductionB_Block_branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels1, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1,
                             out_channels=out_channels1, kernel_size=3, stride=2)
        )
        self.InceptionV4_ReductionB_Block_branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,
                             out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
                                          out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings=[0, 3]),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
                                          out_channels=out_channels2, kernel_sizes=[7, 1], paddings=[3, 0]),
            ConvBNReLU(in_channels=out_channels2,
                             out_channels=out_channels2, kernel_size=3, stride=2)
        )
        self.InceptionV4_ReductionB_Block_branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),  # 20200715 去掉padding = 1
        )
    def forward(self, x):
        out1 = self.InceptionV4_ReductionB_Block_branch1(x)
        out2 = self.InceptionV4_ReductionB_Block_branch2(x)
        out3 = self.InceptionV4_ReductionB_Block_branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

# InceptionV4
class InceptionV4(nn.Module):
    input_resize = (299, 299, 3)
    def __init__(self, num_classes=1000):
        super(InceptionV4, self).__init__()
        self.stem_block = InceptionV4_Stem_Block()
        self.A4_block = nn.Sequential(
            InceptionV2_ModuleA_Block(in_channels=384, out_channels1=96, out_channels2reduce=64,
                                            out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=96),
            InceptionV2_ModuleA_Block(in_channels=384, out_channels1=96, out_channels2reduce=64,
                                            out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=96),
            InceptionV2_ModuleA_Block(in_channels=384, out_channels1=96, out_channels2reduce=64,
                                            out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=96),
            InceptionV2_ModuleA_Block(in_channels=384, out_channels1=96, out_channels2reduce=64,
                                            out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=96),
            InceptionV4_ReductionA_Block(in_channels=384, out_channels1=384,
                                               out_channels2reduceA=192, out_channels2reduceB=224, out_channels2=256)
        )
        self.B7_block = nn.Sequential(
            InceptionV2_ModuleB_Block(in_channels=1024, out_channels1=384, out_channels2reduce=192,
                                            out_channels2=256, out_channels3reduce=192, out_channels3=256, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=1024, out_channels1=384, out_channels2reduce=192,
                                            out_channels2=256, out_channels3reduce=192, out_channels3=256, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=1024, out_channels1=384, out_channels2reduce=192,
                                            out_channels2=256, out_channels3reduce=192, out_channels3=256, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=1024, out_channels1=384, out_channels2reduce=192,
                                            out_channels2=256, out_channels3reduce=192, out_channels3=256, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=1024, out_channels1=384, out_channels2reduce=192,
                                            out_channels2=256, out_channels3reduce=192, out_channels3=256, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=1024, out_channels1=384, out_channels2reduce=192,
                                            out_channels2=256, out_channels3reduce=192, out_channels3=256, out_channels4=128),
            InceptionV2_ModuleB_Block(in_channels=1024, out_channels1=384, out_channels2reduce=192,
                                            out_channels2=256, out_channels3reduce=192, out_channels3=256, out_channels4=128),
            InceptionV4_ReductionB_Block(
                in_channels=1024, out_channels1=192, out_channels2reduce=256, out_channels2=320)
        )
        self.C3_block = nn.Sequential(
            InceptionV2_ModuleC_Block(in_channels=1536, out_channels1=256,
                                            out_channels2reduce=384, out_channels2=256, out_channels3reduce=448, out_channels3=256, out_channels4=256),
            InceptionV2_ModuleC_Block(in_channels=1536, out_channels1=256,
                                            out_channels2reduce=384, out_channels2=256, out_channels3reduce=448, out_channels3=256, out_channels4=256),
            InceptionV2_ModuleC_Block(in_channels=1536, out_channels1=256,
                                            out_channels2reduce=384, out_channels2=256, out_channels3reduce=448, out_channels3=256, out_channels4=256)
        )
        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)
    def forward(self, x):
        x = self.stem_block(x)
        x = self.A4_block(x)
        x = self.B7_block(x)
        x = self.C3_block(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
