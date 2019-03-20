
"""
padding is for image size (104, 80)
(obtained by downsample=2, crop_last_row=True)
"""


cnn_specs = dict()

# standard "small"
# 900k params, 3.4 MB, GPU time: 9
# output size: (12, 9)
spec = dict(
    conv_filter_sizes=[8, 4],
    conv_filters=[16, 32],
    conv_strides=[4, 2],
    conv_pads=[(0, 0), (1, 1)],
    hidden_sizes=[256],
)
cnn_specs["0"] = cnn_specs[0] = spec

# standard NIPS "large"
# 3.6M params, 13.8 MB, GPU time: 14
# output size: (12, 9)
spec = dict(
    conv_filter_sizes=[8, 4, 3],
    conv_filters=[32, 64, 64],
    conv_strides=[4, 2, 1],
    conv_pads=[(0, 0), (1, 1), (1, 1)],
    hidden_sizes=[512],
)
cnn_specs["1"] = cnn_specs[1] = spec

# previous best (used to be #7)
# 2.1M params, 8.0MB, GPU time: 27
# output size: (17, 13)
spec = dict(
    conv_filter_sizes=[5, 3, 3, 3, 3],
    conv_filters=[32, 64, 64, 128, 128],
    conv_strides=[3, 1, 1, 2, 1],
    conv_pads=[(0, 0), (1, 1), (1, 1), (1, 1), (1, 1)],
    hidden_sizes=[64, 64],
)
cnn_specs["2"] = cnn_specs[2] = spec

# new, biggest yet, more resolution throughout conv
# 4.1M params, 15.5 MB, GPU time: 50 (doesn't change if first FCLayer is 32)
# output size: (25, 19)
spec = dict(
    conv_filter_sizes=[4, 3, 3, 3, 3],
    conv_filters=[32, 64, 64, 64, 128],
    conv_strides=[2, 1, 1, 1, 2],
    conv_pads=[(0, 0), (1, 1), (1, 1), (1, 1), (0, 0)],
    hidden_sizes=[64, 64],
)
cnn_specs["3"] = cnn_specs[3] = spec

# new, bigger filters
# 830k params, 3.2 MB, GPU time: 14
# output size: (12, 9)
spec = dict(
    conv_filter_sizes=[16, 8, 4],
    conv_filters=[16, 32, 64],
    conv_strides=[3, 2, 1],
    conv_pads=[(1, 1), (1, 2), (1, 1)],
    hidden_sizes=[256],
)
cnn_specs["4"] = cnn_specs[4] = spec

