import os
import numpy as np
from skimage import io, restoration
from scipy.signal import convolve2d


# User input（从环境变量读取；不内置任何本机默认路径，缺失即报错并打印用法）
input_folder = os.environ.get('SQYE_INPUT_DIR', '').strip()
output_folder = os.environ.get('SQYE_OUTPUT_DIR', '').strip()
if not input_folder or not output_folder:
    raise SystemExit(
        '[sqye] 缺少环境变量 SQYE_INPUT_DIR / SQYE_OUTPUT_DIR'
        '（分别为输入/输出文件夹路径）。\n'
        '用法：先设置环境变量再运行，例如：\n'
        '  PowerShell: $env:SQYE_INPUT_DIR = \'<输入文件夹>\'; '
        '$env:SQYE_OUTPUT_DIR = \'<输出文件夹>\'\n'
        '  cmd:        set SQYE_INPUT_DIR=<输入文件夹> && '
        'set SQYE_OUTPUT_DIR=<输出文件夹>')
saveImage = 1  # 非 0 即保存反卷积结果，0 = 只算不存
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
