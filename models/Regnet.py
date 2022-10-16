from Anynet import AnyNetXe

import numpy as np
import torch
import torch.nn as nn


# Source: https://github.com/rwightman/pytorch-image-models
def _mcfg(**kwargs):
    cfg = dict(se_ratio = 0., bottle_ratio = 1., stem_width=24)
    cfg.update(**kwargs)
    return cfg


# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    # ------RegNetX------
    regnetx_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio = None),
    regnetx_004=_mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, se_ratio = None),
    regnetx_006=_mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16, se_ratio = None),
    regnetx_008=_mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16, se_ratio = None),
    regnetx_016=_mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18, se_ratio = None),
    regnetx_032=_mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25, se_ratio = None),
    regnetx_040=_mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23, se_ratio = None),
    regnetx_064=_mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17, se_ratio = None),
    regnetx_080=_mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23, se_ratio = None),
    regnetx_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio = None),
    regnetx_160=_mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22, se_ratio = None),
    regnetx_320=_mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23, se_ratio = None),
    # ------RegNetY------
    regnety_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    regnety_004=_mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    regnety_006=_mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    regnety_008=_mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    regnety_016=_mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    regnety_032=_mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    regnety_040=_mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    regnety_064=_mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    regnety_080=_mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    regnety_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    regnety_160=_mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    regnety_320=_mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25),
)


def generate_regenet_arch(initial_width, width_slope, quantilize, width_m, depth, bottleneck_ratio, group_width):
    """
    Inspired by the observation from AnyNetXd and AnyNetXe
    """
    assert width_m > 0, 'width_m must be greater than 0'

    # Generate block widths and depths via Eq.2 ~ Eq.4 in the paper
    widths_control = initial_width + np.arange(depth) * width_slope                  # [Eq.2]
    widths_exp = np.round(np.log(widths_control / initial_width) / np.log(width_m))  # [Eq.3]
    widths = initial_width * np.power(width_m, widths_exp)                           # [Eq.4]
    widths = np.round(np.divide(widths, quantilize)) * quantilize

    reg_block_widths, reg_num_blocks = np.unique(widths.astype(int) , return_counts = True)
    reg_group_widths = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in reg_block_widths])
    reg_block_widths = np.round(reg_block_widths // bottleneck_ratio / group_width) * group_width
    reg_group_widths = reg_group_widths * bottleneck_ratio
    reg_bottleneck_ratio = [bottleneck_ratio for _ in range(len(reg_num_blocks))]

    return list(reg_num_blocks), list(reg_block_widths.astype(int)),  list(reg_group_widths), list(reg_bottleneck_ratio)


class RegNet(AnyNetXe):
    def __init__(self, num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn):
        super(RegNet, self).__init__(num_blocks, block_widths, bottleneck_ratios, group_widths, stride, se_ratio, num_classes, sub_stage, use_fpn)


def create_regnet(model_arch = "regnetx_002", stride = 1, num_classes = 1000,
                  sub_stage = ['C', 'C', 'C', 'C'], use_fpn = False):
    model_cfg = model_cfgs[model_arch]
    num_blocks, block_widths, group_widths, bottleneck_ratios = generate_regenet_arch(
        initial_width = model_cfg['w0'],
        width_slope = model_cfg['wa'],
        quantilize = 8,
        width_m = model_cfg['wm'],
        depth = model_cfg['depth'],
        bottleneck_ratio = 1,
        group_width = model_cfg['group_w'],
    )
    model = RegNet(num_blocks, block_widths, bottleneck_ratios, group_widths, stride, model_cfg['se_ratio'], num_classes, sub_stage, use_fpn)
    return model


if __name__ == '__main__':
    model = create_regnet('regnety_040', 2, 1000, sub_stage = ['C', 'C', 'C', 'C'], use_fpn = True)
    # print(model)
    img = torch.randn(1, 3, 224, 224)
    assert model(img).shape == (1, 1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print("RegNet test success!")