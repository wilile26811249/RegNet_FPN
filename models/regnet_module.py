import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerBlock, Transformer


class Stem(nn.Module):
    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels, 3, stride = 2, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels) -> None:
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels = 0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

    def get_result_from_inner_blocks(self, x, idx):
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x, idx):
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x):
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(F.adaptive_avg_pool2d(self.get_result_from_layer_blocks(last_inner, -1),1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            # results.insert(0, last_inner)
            results.insert(0,
                F.adaptive_avg_pool2d(self.get_result_from_layer_blocks(last_inner, idx), 1))
        results = torch.cat(results, dim = 1)
        return results


class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class XBlock(nn.Module):
    """
    The XBlock is based on the standard residual bottleneck block with group convolution.
    """
    def __init__(self, in_channels, out_channels, stride, bottleneck_ratio, group_width, se_ratio = None):
        super(XBlock, self).__init__()
        inner_channels = out_channels // bottleneck_ratio
        inner_groups = inner_channels // group_width
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size = 3, stride = stride, padding = 1, groups = inner_groups, bias = False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace = True)
        )

        if se_ratio is not None:
            se_channels = int(inner_channels * se_ratio)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inner_channels, se_channels, kernel_size = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(se_channels, inner_channels, kernel_size = 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        self.conv3 = nn.Sequential(
            nn.Conv2d(inner_channels, out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        if self.se is not None:
            se = self.se(x1)
            x1 = x1 * se
        x1 = self.conv3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.relu(x1 + x2)
        return x


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride, bottleneck_ratio, group_width, se_ratio = None):
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()

        # Each stage consists of a sequence of identical blocks,
        # except for the first block which use stride-two conv.
        self.blocks.add_module("block_0", XBlock(in_channels, out_channels, stride, bottleneck_ratio, group_width, se_ratio))
        for index in range(1, num_blocks):
            self.blocks.add_module(f"block_{index + 1}", XBlock(out_channels, out_channels, 1, bottleneck_ratio, group_width, se_ratio))

    def forward(self, x):
        x = self.blocks(x)
        return x


class TransformerStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, imagesize, downsample = True):
        super(TransformerStage, self).__init__()
        self.blocks = nn.Sequential()

        # Each stage consists of a sequence of identical blocks,
        # except for the first block which use stride-two conv.
        # self.blocks.add_module("block_0", TransformerBlock(in_channels, out_channels, stride))
        # for index in range(1, num_blocks):
        #     self.blocks.add_module(f"block_{index + 1}", TransformerBlock(out_channels, out_channels, 1))
        ih, iw = imagesize
        self.blocks.add_module("block_0", Transformer(in_channels, out_channels, imagesize, downsample = downsample))
        for index in range(1, num_blocks):
            self.blocks.add_module(f"block_{index + 1}", Transformer(out_channels, out_channels, imagesize))


    def forward(self, x):
        x = self.blocks(x)
        return x


if __name__ == '__main__':
    image = torch.randn(1, 3, 64, 64)
    model = Stage(in_channels = 3, out_channels = 16, num_blocks = 10, stride = 2, bottleneck_ratio = 4, group_width = 4)
    output = model(image)
    assert output.shape == torch.Size([1, 16, 32, 32])
    print("RegNet Module Test Passed!")

    model = TransformerStage(in_channels = 3, out_channels = 16, num_blocks = 3,
                             imagesize = (32, 32), downsample = True)
    output = model(image)
    assert output.shape == torch.Size([1, 16, 32, 32])
    print("RegNet Module Test Passed!")

