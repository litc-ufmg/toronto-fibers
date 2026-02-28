import os
import numpy as np
import tifffile

def load_microct_image(filepath):
    """
    Reads a TIFF file (MicroCT) and converts it to 8-bit grayscale.
    
    The logic follows the pattern from the original notebook:
    1. Reads the raw file.
    2. Normalizes by dividing by the maximum value.
    3. Scales to 0-255 range and converts to uint8.
    
    Args:
        filepath (str): Path to the image file.
        
    Returns:
        np.ndarray: 3D image volume in uint8 format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo não foi encontrado: {filepath}")
    
    raw_tiff_img = tifffile.imread(filepath)
    img_8bit = (raw_tiff_img / np.max(raw_tiff_img) * 255).astype(np.uint8)

    return img_8bit