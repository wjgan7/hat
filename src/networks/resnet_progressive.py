import copy

import torch
import torch.nn as nn

import utils

EXPAND_FACTOR = 2


def new_layer_size(layers, num_tasks):
    return int(EXPAND_FACTOR * layers / num_tasks)


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
        self.conv1 = nn.ModuleList()
        self.V1scale = nn.ModuleList()
        self.V1x1 = nn.ModuleList()
        self.U1 = nn.ModuleList()

        self.conv2 = nn.ModuleList()
        self.V2scale = nn.ModuleList()
        self.V2x1 = nn.ModuleList()
        self.U2 = nn.ModuleList()

        self.bn1 = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.ModuleList()
        if downsample is not None:
            self.downsample = True
            self.convd = nn.ModuleList()
            self.bnd = nn.ModuleList()
            self.VDscale = nn.ModuleList()
            self.VDx1 = nn.ModuleList()
            self.UD = nn.ModuleList()
        else:
            self.downsample = None
        self.stride = stride

        self.num_tasks = num_tasks
        for t in range(self.num_tasks):
            self.conv1.append(conv3x3(inplanes, planes, stride))
            self.conv2.append(conv3x3(planes, planes))
            self.bn1.append(norm_layer(planes))
            self.bn2.append(norm_layer(planes))
            if self.downsample:
                self.convd.append(copy.deepcopy(downsample[0]))
                self.bnd.append(copy.deepcopy(downsample[1]))
            if t > 0:
                self.V1scale.append(nn.Embedding(1, t))
                self.V1x1.append(nn.Conv2d(t * inplanes, inplanes, 1))
                self.U1.append(conv3x3(inplanes, planes, stride))

                self.V2scale.append(nn.Embedding(1, t))
                self.V2x1.append(nn.Conv2d(t * planes, planes, 1))
                self.U2.append(conv3x3(planes, planes))

                if self.downsample is not None:
                    self.VDscale.append(nn.Embedding(1, t))
                    self.VDx1.append(nn.Conv2d(t * inplanes, inplanes, 1))
                    self.UD.append(copy.deepcopy(downsample[0]))

    def forward(self, x, t, prev):
        identity = x
        identity_prev = prev

        out = self.conv1[t](x)
        if t > 0:
            scaled = [self.V1scale[t-1].weight[0][j] * prev[j] for j in range(t)]
            concatenated = torch.cat(scaled, 1)
            projected = self.V1x1[t-1](concatenated)
            activations = self.relu(projected)
            adapted = self.U1[t-1](activations)
            out += adapted
            prev = [self.relu(self.bn1[j](self.conv1[j](prev[j]))) for j in range(t)]
        out = self.bn1[t](out)
        out = self.relu(out)

        out = self.conv2[t](out)
        if t > 0:
            scaled = [self.V2scale[t-1].weight[0][j] * prev[j] for j in range(t)]
            concatenated = torch.cat(scaled, 1)
            projected = self.V2x1[t-1](concatenated)
            activations = self.relu(projected)
            adapted = self.U2[t-1](activations)
            out += adapted
            prev = [self.bn2[j](self.conv2[j](prev[j])) for j in range(t)]
        out = self.bn2[t](out)

        if self.downsample is not None:
            identity = self.convd[t](identity)
            if t > 0:
                scaled = [self.VDscale[t-1].weight[0][j] * identity_prev[j] for j in range(t)]
                concatenated = torch.cat(scaled, 1)
                projected = self.VDx1[t-1](concatenated)
                activations = self.relu(projected)
                adapted = self.UD[t-1](activations)
                identity += adapted
                identity_prev = [self.bnd[j](self.convd[j](identity_prev[j])) for j in range(t)]
            identity = self.bnd[t](identity)
        if t > 0:
            prev = [self.relu(prev[j] + identity_prev[j]) for j in range(t)]
        out += identity
        out = self.relu(out)

        return out, prev

    def unfreeze_column(self, t):
        for i in range(self.num_tasks):
            requires_grad = (i == t)
            utils.set_req_grad(self.conv1[i], requires_grad)
            utils.set_req_grad(self.conv2[i], requires_grad)
            utils.set_req_grad(self.bn1[i], requires_grad)
            utils.set_req_grad(self.bn2[i], requires_grad)
            if self.downsample is not None:
                utils.set_req_grad(self.convd[i], requires_grad)
                utils.set_req_grad(self.bnd[i], requires_grad)
            if i > 0:
                utils.set_req_grad(self.V1scale[i-1], requires_grad)
                utils.set_req_grad(self.V1x1[i-1], requires_grad)
                utils.set_req_grad(self.U1[i-1], requires_grad)
                utils.set_req_grad(self.V2scale[i-1], requires_grad)
                utils.set_req_grad(self.V2x1[i-1], requires_grad)
                utils.set_req_grad(self.U2[i-1], requires_grad)
                if self.downsample is not None:
                    utils.set_req_grad(self.VDscale[i-1], requires_grad)
                    utils.set_req_grad(self.VDx1[i-1], requires_grad)
                    utils.set_req_grad(self.UD[i-1], requires_grad)
                    utils.set_req_grad(self.UD[i-1], requires_grad)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, taskcla=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_tasks = len(taskcla)

        self.inplanes = new_layer_size(64, self.num_tasks)
        original_inplanes = self.inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       new_layer_size(64, self.num_tasks),
                                       layers[0])
        self.layer2 = self._make_layer(block,
                                       new_layer_size(128, self.num_tasks),
                                       layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       new_layer_size(256, self.num_tasks),
                                       layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       new_layer_size(512, self.num_tasks),
                                       layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = new_layer_size(512 * block.expansion, self.num_tasks)
        self.last = nn.ModuleList()
        self.Vflscale = nn.ModuleList()
        self.Vfl = nn.ModuleList()
        self.Ufl = nn.ModuleList()
        for t, n in taskcla:
            self.conv1.append(nn.Conv2d(3, original_inplanes, kernel_size=7,
                                        stride=2, padding=3, bias=False))
            self.bn1.append(norm_layer(original_inplanes))
            self.last.append(torch.nn.Linear(in_features, n))
            if t > 0:
                self.Vflscale.append(nn.Embedding(1, t))
                self.Vfl.append(nn.Linear(t * in_features, in_features))
                self.Ufl.append(nn.Linear(in_features, n))

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
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

    def _forward_impl(self, x, t):
        # See note [TorchScript super()]

        prev = None
        if t > 0:
            prev = [self.maxpool(self.relu(self.bn1[j](self.conv1[j](x))))
                    for j in range(t)]
        x = self.conv1[t](x)
        x = self.bn1[t](x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, prev = self.layer1[0](x, t, prev)
        x, prev = self.layer1[1](x, t, prev)
        x, prev = self.layer2[0](x, t, prev)
        x, prev = self.layer2[1](x, t, prev)
        x, prev = self.layer3[0](x, t, prev)
        x, prev = self.layer3[1](x, t, prev)
        x, prev = self.layer4[0](x, t, prev)
        x, prev = self.layer4[1](x, t, prev)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.last[t](x)
        if t > 0:
            prev = [torch.flatten(self.avgpool(prev[j]), 1) for j in range(t)]
            # sum laterals, no non-linearity for last layer
            scaled = [self.Vflscale[t-1].weight[0][j] * prev[j] for j in range(t)]
            concatenated = torch.cat(scaled, 1)
            projected = self.Vfl[t-1](concatenated)
            adapted = self.Ufl[t-1](projected)
            x += adapted
        y = [None for _ in range(self.num_tasks)]
        y[t] = x
        return y

    def forward(self, x, t):
        return self._forward_impl(x, t)

    #train only the current column subnet
    def unfreeze_column(self, t):
        self.layer1[0].unfreeze_column(t)
        self.layer1[1].unfreeze_column(t)
        self.layer2[0].unfreeze_column(t)
        self.layer2[1].unfreeze_column(t)
        self.layer3[0].unfreeze_column(t)
        self.layer3[1].unfreeze_column(t)
        self.layer4[0].unfreeze_column(t)
        self.layer4[1].unfreeze_column(t)
        for i in range(self.num_tasks):
            requires_grad = (i == t)
            utils.set_req_grad(self.conv1[i], requires_grad)
            utils.set_req_grad(self.bn1[i], requires_grad)
            utils.set_req_grad(self.last[i], requires_grad)
            if i > 0:
                utils.set_req_grad(self.Vflscale[i-1], requires_grad)
                utils.set_req_grad(self.Vfl[i-1], requires_grad)
                utils.set_req_grad(self.Ufl[i-1], requires_grad)


def Net(inputsize, taskcla):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla=taskcla)
