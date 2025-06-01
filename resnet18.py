import torch as t

# This implementation was copied from the base pytorch one with some modifications:
# - unused features were stripped out (dilation, etc.)
# - the model uses preactivation conv to match the deep double descent paper (bn -> relu -> conv)
# - removed maxpool because it seems like it downsamples our 32x32 cifar images too early

class BasicBlock(t.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        self.relu = t.nn.ReLU(inplace=True)

        self.conv1 = t.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = t.nn.BatchNorm2d(inplanes)

        self.conv2 = t.nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = t.nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = t.nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        # See: https://arxiv.org/pdf/1603.05027
        # and "BatchNorm-ReLU-Convoultion" in https://arxiv.org/pdf/1912.02292
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out

class ResNet(t.nn.Module):
    def __init__(
        self,
        layers: list[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        k: int = 64,
    ) -> None:
        super().__init__()
        self.inplanes = k

        self.conv1 = t.nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(k, layers[0])
        self.layer2 = self._make_layer(k*2, layers[1], stride=2)
        self.layer3 = self._make_layer(k*4, layers[2], stride=2)
        self.layer4 = self._make_layer(k*8, layers[3], stride=2)
        self.avgpool = t.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = t.nn.Linear(k*8, num_classes)

        for m in self.modules():
            if isinstance(m, t.nn.Conv2d):
                t.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (t.nn.BatchNorm2d, t.nn.GroupNorm)):
                t.nn.init.constant_(m.weight, 1)
                t.nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> t.nn.Sequential:
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return t.nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = t.flatten(x, 1)
        x = self.fc(x)

        return x

