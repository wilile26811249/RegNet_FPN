from regnet_module import Stem, Stage, TransformerStage, FeaturePyramidNetwork, Head

import torch
import torch.nn as nn


class AnyNetX(nn.Module):
    """
    Initial AnyNet design space.

    Consists of a simple stem, followed by the network body that performs the
    bulk of the computation, and a final network head that predicts the classes.
    """
    def __init__(self, imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn):
        super(AnyNetX, self).__init__()
        self.use_fpn = use_fpn
        prev_conv_width = 32
        # Check the input parameters are valid
        for block_width, bottleneck_ratio, group_width, stage_type in zip(block_widths, bottleneck_ratios, group_widths, sub_stage):
            assert block_width % (bottleneck_ratio * group_width) == 0
        # Construct the stem
        self.stem = Stem(prev_conv_width)
        # Construct the body
        self.body = nn.Sequential()
        for index, (block_width, num_block, bottleneck_ratio, group_width, stage_type) in enumerate(zip(block_widths, num_blocks, bottleneck_ratios, group_widths, sub_stage)):
            if stage_type == 'C':
                stage = Stage(prev_conv_width, block_width, num_block, stride, bottleneck_ratio, group_width, se_ratio)
            else:
                ih, iw = imagesize
                ih = ih // (2**(index + 2))
                iw = iw // (2**(index + 2))
                stage = TransformerStage(prev_conv_width, block_width, num_block, (ih, iw), downsample = (stride == 2))
            self.body.add_module(f"Stage_{index + 1}", stage)
            prev_conv_width = block_width
        # Construct the FPN
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork(block_widths, prev_conv_width // 4)
            self.fpn_input = []
        # Construct the head
        self.head = Head(prev_conv_width, num_classes)

    def forward(self, x):
        x = self.stem(x)
        if self.use_fpn:
            for index in range(len(self.body)):
                x = self.body[index](x)
                self.fpn_input.append(x)
            x = self.fpn(self.fpn_input)
        else:
            x = self.body(x)
        x = self.head(x)
        return x


class AnyNetXb(AnyNetX):
    """
    AnyNetXb is a variant of AnyNetX that uses the same bottleneck ratio for all.
    """
    def __init__(self, imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn):
        super(AnyNetXb, self).__init__(imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn)
        assert len(set(bottleneck_ratios)) == 1, "All bottleneck ratios must be equal"


class AnyNetXc(AnyNetXb):
    """
    AnyNetXc is a variant of AnyNetXb that uses the same group width for all.
    """
    def __init__(self, imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn):
        super(AnyNetXc, self).__init__(imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn)
        assert len(set(group_widths)) == 1, "All group widths must be equal"


class AnyNetXd(AnyNetXc):
    """
    AnyNetXd is a variant of AnyNetXc that block widths are monotonically increasing.
    """
    def __init__(self, imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn):
        super(AnyNetXd, self).__init__(imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn)
        assert all(prev <= behind for prev, behind in zip(block_widths[: -2], block_widths[1 :])), "Block widths must be monotonically increasing"


class AnyNetXe(AnyNetXd):
    """
    AnyNetXe is a variant of AnyNetXd that number of blocks is monotonically increasing.
    """
    def __init__(self, imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn):
        super(AnyNetXe, self).__init__(imagesize, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn)
        assert all(prev <= behind for prev, behind in zip(num_blocks[: -2], num_blocks[1 :])), "Number of blocks must be monotonically increasing"


if __name__ == '__main__':
    num_blocks = [1, 2, 7, 12]
    block_widths = [32, 64, 160, 384]
    bottleneck_ratios = [1, 1, 1, 1]
    group_widths = [2, 2, 2, 2]
    model = AnyNetXd(num_blocks, block_widths, bottleneck_ratios,
        group_widths, stride = 1, se_ratio = 4, num_classes = 10)
    print(model)
    img = torch.randn(1, 3, 224, 224)
    print(model(img).shape)
    print(sum(p.numel() for p in model.parameters()))
