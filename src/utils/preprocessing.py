import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

def auto_crop_volume(volume, threshold=0, padding=0, *, verbose=True):
    """
    Automatically crops the 3D volume to remove empty (black) margins.
    
    It finds the bounding box of the object (pixels > threshold) and slices
    the volume to keep only the region of interest.
    
    Args:
        volume (np.ndarray): Input 3D image volume.
        threshold (int/float): Intensity value to consider as background (default 0).
        padding (int): Number of extra pixels to keep around the object (default 0).
        verbose (bool): Print crop details. Empty-volume warnings remain visible.
        
    Returns:
        np.ndarray: Cropped volume.
    """
    # Create a boolean mask where the object is present
    mask = volume > threshold

    # If the image is completely empty, return it as is
    if not np.any(mask):
        print("Warning: Volume is empty (all pixels <= threshold). Returning original.")
        return volume

    # Find the indices of the object along each axis
    # We use np.any to project the mask, which is much faster than np.argwhere on large 3D arrays
    z_any = np.any(mask, axis=(1, 2))
    y_any = np.any(mask, axis=(0, 2))
    x_any = np.any(mask, axis=(0, 1))

    # Get the first and last index where the object appears
    z_min, z_max = np.where(z_any)[0][[0, -1]]
    y_min, y_max = np.where(y_any)[0][[0, -1]]
    x_min, x_max = np.where(x_any)[0][[0, -1]]

    # Apply padding and ensure we don't go out of bounds
    z_min = max(0, z_min - padding)
    z_max = min(volume.shape[0], z_max + 1 + padding)

    y_min = max(0, y_min - padding)
    y_max = min(volume.shape[1], y_max + 1 + padding)

    x_min = max(0, x_min - padding)
    x_max = min(volume.shape[2], x_max + 1 + padding)

    if verbose:
        print(
            f"      Crop (Z, Y, X): {volume.shape} -> "
            f"{(z_max-z_min, y_max-y_min, x_max-x_min)}",
            flush=True,
        )

    # Return the sliced volume
    return volume[z_min:z_max, y_min:y_max, x_min:x_max]

def binarize_volume(volume, plot_hist=False):
    """
    Binarizes the 3D image using otsu criterium.

    Args:
        volume (np.ndarray): 3D image to be binarized.
        plot_hist (bool): Indicates if the histogram of the pixels values and the threshold should be plotted.
    """
    # Choosing threshold with Otsu criterium
    binary_threshold = threshold_otsu(volume)

    if plot_hist:
        # Verifying the chosen threshold looking at the voxels intensity distributions
        histogram_intensity, bins = np.histogram(volume.ravel(), bins=256, range=(0, 256))

        plt.figure(figsize=(10, 6))
        plt.title("Histograma do Volume 3D")
        plt.xlabel("Nível de Cinza (Intensidade do Voxel)")
        plt.ylabel("Frequência")
        plt.plot(histogram_intensity, color='gray')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axvline(x=binary_threshold, color='r', linestyle='--')
        plt.show()

    binary_volume = volume > binary_threshold
    return binary_volume

def select_cubic_section(volume, pos='ct', n_pixels=400):
    """
    Selects a cubic section of the 3D image for testing purposes.
    
    Based on the coordinate system:
    - Top-Left (tl) starts at [0, 0, 0]
    - Bottom-Right (br) ends at [z_max, y_max, x_max]

    Args:
        volume (np.ndarray): Input 3D image volume (Z, Y, X).
        pos (str): Position of the cubic crop. Options:
            'ct': Center (middle of Z, Y, X).
            'tl': Top-Left (Start of Z, Y, X).
            'tr': Top-Right (Start of Z, Y; End of X).
            'bl': Bottom-Left (End of Z, Y; Start of X).
            'br': Bottom-Right (End of Z, Y, X).
        n_pixels (int): Size of the cube edge in pixels (default: 400).
    
    Returns:
        np.ndarray: The cropped cubic volume.
    """
    # Get volume dimensions (Depth/Z, Height/Y, Width/X)
    d, h, w = volume.shape
    
    # Check if the requested size is larger than the image
    if n_pixels > min(d, h, w):
        print(f"Warning: n_pixels ({n_pixels}) is larger than one of the dimensions {volume.shape}. Returning original.")
        return volume

    # Calculate starting indices based on position
    if pos == 'ct':
        z_s = (d - n_pixels) // 2
        y_s = (h - n_pixels) // 2
        x_s = (w - n_pixels) // 2
        
    elif pos == 'tl': # Top-Left (Start of array)
        z_s, y_s, x_s = 0, 0, 0
        
    elif pos == 'br': # Bottom-Right (End of array)
        z_s = d - n_pixels
        y_s = h - n_pixels
        x_s = w - n_pixels
        
    elif pos == 'tr': # Top-Right (Start Z/Y, End X)
        z_s = 0
        y_s = 0
        x_s = w - n_pixels
        
    elif pos == 'bl': # Bottom-Left (End Z/Y, Start X)
        z_s = d - n_pixels
        y_s = h - n_pixels
        x_s = 0
        
    else:
        raise ValueError(f"Invalid position '{pos}'. Use 'ct', 'tl', 'tr', 'bl', or 'br'.")

    # Define ending indices
    z_e = z_s + n_pixels
    y_e = y_s + n_pixels
    x_e = x_s + n_pixels

    print(f"Cropping cubic section at '{pos}': Z[{z_s}:{z_e}], Y[{y_s}:{y_e}], X[{x_s}:{x_e}]")
    
    return volume[z_s:z_e, y_s:y_e, x_s:x_e]
