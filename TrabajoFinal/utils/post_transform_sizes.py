def post_conv_shape(
    h_in,
    w_in,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1
):
    h_out = int(((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    w_out = int(((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return out_channels, h_out, w_out


def post_transconv_shape(
    h_in,
    w_in,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1
):
    h_out = (h_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    w_out = (w_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return out_channels, h_out, w_out


def post_maxpool_shape(
    h_in,
    w_in,
    out_channels,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1
):
    if stride is None:
        stride = kernel_size
    h_out = int(((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    w_out = int(((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return out_channels, h_out, w_out