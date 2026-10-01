import os
import numpy as np
from skimage import io, restoration
from scipy.signal import convolve2d


# User input
input_folder = r'D:/data/input'  # 输入文件夹路径
output_folder = r'D:/data/output'  # 输出文件夹路径
saveImage = 5  # 1 = save images, 0 = don't save images
psf_radius = 2
nsr = 0.6  # noise to signal ratio

# Create output folder if needed
if saveImage:
    os.makedirs(output_folder, exist_ok=True)


# Create PSF (Point Spread Function) - 'disk' equivalent
def create_psf(radius, scale=1):
    # Create meshgrid for PSF computation
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    x = scale * x
    y = scale * y
    # Compute sinc function-based PSF
    r = np.sqrt(x ** 2 + y ** 2)
    indices = np.argwhere(r == 0)
    r[indices] = 1
    psf = np.sin(r) / r
    psf[indices] = 1
    psf[np.isnan(psf)] = 1  # Handle NaN values
    psf = psf / np.sum(psf)  # Normalize the PSF
    return psf


# Create PSF with a radius of 3
psf = create_psf(3)

# Process images
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)

    # Check if it's a valid file (e.g., check for tif or png files)
    if os.path.isfile(file_path) and file_name.endswith('.tif'):
        # Read the image
        image = io.imread(file_path)

        # Perform Wiener Deconvolution
        deconv_image = restoration.wiener(image, psf, balance=nsr)

        # Save the deconvolved image if saveImage is 1
        if saveImage:
            output_path = os.path.join(output_folder, file_name)
            io.imsave(output_path, deconv_image.astype(np.uint8))  # Save as uint8 image
