'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ensembles_v2.BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type = "float"):
        super(BasicBlock, self).__init__()
        if block_type == "binary":
            ConvLayer = BinaryConv2d
            LinearLayer = BinaryLinear
            Activation = BinaryTanh
        else:
            ConvLayer = nn.Conv2d
            LinearLayer = nn.Linear
            Activation = nn.ReLU

        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.a1 = Activation()

        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.a2 = Activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                ConvLayer(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.a2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, block_type = "float"):
        super(Bottleneck, self).__init__()

        if block_type == "binary":
            ConvLayer = BinaryConv2d
            LinearLayer = BinaryLinear
            Activation = BinaryTanh
        else:
            ConvLayer = nn.Conv2d
            LinearLayer = nn.Linear
            Activation = nn.ReLU

        self.conv1 = ConvLayer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.a1 = Activation()

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = ConvLayer(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.a2 = Activation()
        self.a3 = Activation()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                ConvLayer(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.a2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.a3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, model_type = "float"):
        super(ResNet, self).__init__()

        if model_type == "binary":
            ConvLayer = BinaryConv2d
            LinearLayer = BinaryLinear
            Activation = BinaryTanh
        else:
            ConvLayer = nn.Conv2d
            LinearLayer = nn.Linear
            Activation = nn.ReLU

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, model_type = model_type)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, model_type = model_type) # 2048 => 16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, model_type = model_type) # 1024
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, model_type = model_type) # 512
        self.a1 = Activation()
        self.linear = LinearLayer(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, model_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, model_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet11():
    return ResNet(BasicBlock, [1,1,1,1])

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])