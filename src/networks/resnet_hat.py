import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_tasks=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # HAT code
        self.ec1 = nn.Embedding(num_tasks, planes)
        self.ec2 = nn.Embedding(num_tasks, planes)

    def mask(self, t, s=1):
        gc1 = torch.sigmoid(s * self.ec1(t))
        gc2 = torch.sigmoid(s * self.ec2(t))
        return [gc1, gc2]

    def forward(self, x, masks):
        identity = x

        # HAT code
        gc1, gc2 = masks

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # HAT code
        out = out * gc1.view(1, -1, 1, 1).expand_as(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # HAT code
        out = out * gc2.view(1, -1, 1, 1).expand_as(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, taskcla=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # HAT code
        self.num_tasks = len(taskcla)

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.last = nn.ModuleList()
        for _, n in taskcla:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                    # nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # HAT code
        self.ec1 = nn.Embedding(self.num_tasks, 64)
        self.parameter_dict = {n: p for n, p in self.named_parameters()}
        self.pre_mask = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            self.num_tasks))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, num_tasks=self.num_tasks))

        return nn.Sequential(*layers)

    def mask(self, t, s=1):
        gc1 = torch.sigmoid(s * self.ec1(t))
        layer_masks = sum([
            self.layer1[0].mask(t, s),
            self.layer1[1].mask(t, s),
            self.layer2[0].mask(t, s),
            self.layer2[1].mask(t, s),
            self.layer3[0].mask(t, s),
            self.layer3[1].mask(t, s),
            self.layer4[0].mask(t, s),
            self.layer4[1].mask(t, s),
        ], [])
        return [gc1] + layer_masks

    def _forward_impl(self, t, x, s=1):
        # See note [TorchScript super()]

        # HAT code
        masks = self.mask(t, s)
        gc1 = masks[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # HAT code
        x = x * gc1.view(1, -1, 1, 1).expand_as(x)

        x = self.layer1[0](x, masks[1:3])
        x = self.layer1[1](x, masks[3:5])
        x = self.layer2[0](x, masks[5:7])
        x = self.layer2[1](x, masks[7:9])
        x = self.layer3[0](x, masks[9:11])
        x = self.layer3[1](x, masks[11:13])
        x = self.layer4[0](x, masks[13:15])
        x = self.layer4[1](x, masks[15:17])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = []
        for i in range(self.num_tasks):
            y.append(self.last[i](x))

        return y, masks

    def forward(self, t, x, s=1):
        return self._forward_impl(t, x, s)

    def get_view_for(self, n, masks):
        gc1 = masks[0]
        if n == 'conv1.weight':
            self.pre_mask = gc1
            return gc1.data.view(-1, 1, 1, 1).expand_as(self.parameter_dict[n])
        elif n == 'conv1.bias':
            return gc1.data.view(-1)

        if n.startswith('layer') and 'conv' in n:
            layer = int(n[5]) - 1
            seq = int(n[7])
            conv = int(n[13]) - 1
            # print(n, layer, seq, conv)
            gc = masks[layer * 4 + seq * 2 + conv + 1]
            if 'weight' in n:
                post = gc.data.view(-1, 1, 1, 1).expand_as(self.parameter_dict[n])
                pre = self.pre_mask.data.view(1, -1, 1, 1).expand_as(self.parameter_dict[n])
                self.pre_mask = gc
                return torch.min(post, pre)
            else:
                return gc.data.view(-1)

        return None


def Net(inputsize, taskcla):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla=taskcla)
