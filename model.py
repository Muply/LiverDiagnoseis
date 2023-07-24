import torch
import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, concat=False):
        super(Encoder, self).__init__()
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())

        self.layer1 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[0],
            self.base_layers[1],
            self.base_layers[2],
        )
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]

    def forward(self, input):
        e1 = self.layer1(input)  # 64,112,112 64
        e2 = self.layer2(e1)  # 64,56,56 256
        e3 = self.layer3(e2)  # 128,28,28 512
        e4 = self.layer4(e3)  # 256,14,14 1024
        f = self.layer5(e4)  # 512,7,7 2048
        return f

# DoubleConv
class DoubleConv(nn.Module):
    # (conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.channel_conv = nn.Sequential(
            nn.BatchNorm2d(in_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        self._init_weight()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# SENet通道注意力
class ChannelAttention_SE(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(avg_out)

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)

# 主干网络
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.encoder_1 = Encoder()
        self.encoder_2 = Encoder()
        self.encoder_3 = Encoder()
        self.encoder_4 = Encoder()
        self.conv1 = DoubleConv(4096, 2048)
        self.conv2 = DoubleConv(2048, 1024)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.att1 = ChannelAttention_SE(4096)
        self.satt1 = SpatialAttention(3)
        self.satt2 = SpatialAttention(3)
        self.satt3 = SpatialAttention(3)
        self.fc = nn.Linear(1024, 2)

    def forward(self, imgs):
        # print(imgs.shape) # torch.Size([128, 24, 224, 224])
        # imm = torch.split(imgs, 3, 1)
        # print(imm)
        # print(imm.shape())
        im1, im2, im3, im4, im5, im6, im7, im8 = torch.split(imgs, 3, 1)
        # 分别提特征
        im1 = self.encoder_1(im1)  # 512 7 7
        im2 = self.encoder_1(im2)  # 512 7 7
        im3 = self.encoder_1(im3)  # 512 7 7
        im4 = self.encoder_2(im4)  # 512 7 7
        im5 = self.encoder_2(im5)  # 512 7 7
        im6 = self.encoder_3(im6)  # 512 7 7
        im7 = self.encoder_3(im7)  # 512 7 7
        im8 = self.encoder_4(im8)  # 512 7 7
        # 拼接
        im1 = torch.cat((im1, im2, im3), 1)  # 1536 7 7
        im4 = torch.cat((im4, im5), 1)  # 1024 7 7
        im6 = torch.cat((im6, im7), 1)  # 1024 7 7


        im1_att = self.satt1(im1)
        im1 = im1 * im1_att
        im4_att = self.satt2(im4)
        im4 = im4 * im4_att
        im6_att = self.satt3(im6)
        im6 = im6 * im6_att

        x = torch.cat((im1, im4, im6, im8), 1)  # 4096 7 7

        # DoubleConv
        att = self.att1(x)
        x *= att
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        # print(type(x))
        #
        # print(x.shape)    # torch.Size([128, 1024])
        # return
        x = self.fc(x)
        return x

