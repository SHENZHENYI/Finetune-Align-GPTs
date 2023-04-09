import torch
import torch.nn.functional as F

# Input tensor with shape [batch_size=1, channels, sequence_length]
input_tensor = torch.randn(1, 6, 10)
# Convolutional filter with shape [output_channels, input_channels, kernel_width=1]
conv_filter = torch.randn(4, 6, 1)

# Perform 1-dimensional convolution
conv_result = F.conv1d(input_tensor, conv_filter, groups=6)

print(conv_result.shape)