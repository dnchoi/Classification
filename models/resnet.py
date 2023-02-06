import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        input_ch,
        out_ch,
        stride=1,
    ):
        super().__init__()

        # residual function BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정
        self.residual_func = nn.Sequential(
            nn.Conv2d(
                in_channels=input_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_ch * BasicBlock.expansion),
        )

        # shortcut, identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용
        self.shortcut = nn.Sequential()

        # projection mapping using 1x1 conv
        if stride != 1 or input_ch != BasicBlock.expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_ch,
                    out_channels=out_ch * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_ch * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_func(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, input_ch, output_ch, stride=1):
        super().__init__()
        self.residual_func = nn.Sequential(
            nn.Conv2d(
                in_channels=input_ch,
                out_channels=output_ch,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=output_ch,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=output_ch,
                out_channels=output_ch,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=output_ch,
                out_channels=output_ch * BottleNeck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=output_ch * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or input_ch != BottleNeck.expansion * output_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_ch,
                    out_channels=output_ch * BottleNeck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=output_ch * BottleNeck.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_func(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, class_number=100):
        super().__init__()
        self.input_channels = 64
        self.conv1_x = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.conv2_x = self._make_layer(block=block, output_ch=64, num_blocks=num_blocks[0], stride=1)
        self.conv3_x = self._make_layer(block=block, output_ch=128, num_blocks=num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(block=block, output_ch=256, num_blocks=num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(block=block, output_ch=512, num_blocks=num_blocks[3], stride=2)

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fully_connected = nn.Linear(512 * block.expansion, class_number)

    def _make_layer(self, block, output_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.input_channels, output_ch, s))
            self.input_channels = output_ch * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1_x(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.average_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)

        return out


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])
