from jax import random
import jax.numpy as jnp
from jax.lax import conv_general_dilated

# Create a random key
key = random.PRNGKey(42)

# Create your 3D arrays
input_array = random.uniform(key, (2, 1000, 3))  # shape: (batch_size, height, channels)
kernel_array = random.uniform(key, (1, 2, 1))  # shape: (filter_height, filter_width, input_channels)

# Specify dilation and strides
dilation = (1, 1, 1)
strides = (1, 1, 1)

# Perform 3D convolution along the height axis (axis=1)
output_array = conv_general_dilated(input_array, kernel_array, window_strides=strides, padding="VALID", dimension_numbers=('NHWC', 'HWIO', 'NHWC'), rhs_dilation=dilation)

# Print the shape of the output array
print("Output shape:", output_array.shape)
