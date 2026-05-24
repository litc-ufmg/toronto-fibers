import numpy as np
from skimage.filters import threshold_otsu


def select_center_cube(volume, edge_length):
    """
    Extracts a cubic region of the given edge length from the geometric center
    of a 3D volume.

    Args:
        volume (np.ndarray): Input 3D volume with shape (Z, Y, X).
        edge_length (int): Cube edge length in voxels.

    Returns:
        np.ndarray: Cubic sub-volume of shape (edge_length, edge_length, edge_length).
                    If the requested edge is larger than any axis, that axis is
                    clipped to the full extent of the volume.
    """
    d, h, w = volume.shape

    if edge_length > min(d, h, w):
        print(
            f"Warning: edge_length ({edge_length}) is larger than one of the "
            f"dimensions {volume.shape}. The cube will be clipped to fit."
        )

    z_size = min(edge_length, d)
    y_size = min(edge_length, h)
    x_size = min(edge_length, w)

    z_s = (d - z_size) // 2
    y_s = (h - y_size) // 2
    x_s = (w - x_size) // 2

    z_e = z_s + z_size
    y_e = y_s + y_size
    x_e = x_s + x_size

    print(
        f"Cropping center cube: Z[{z_s}:{z_e}], Y[{y_s}:{y_e}], X[{x_s}:{x_e}]"
    )

    return volume[z_s:z_e, y_s:y_e, x_s:x_e]


def binarize_volume_otsu(volume, plot_hist=False):
    """
    Binarizes a 3D volume using Otsu's global thresholding criterion.

    Args:
        volume (np.ndarray): Input 3D grayscale volume.
        plot_hist (bool): If True, plots the intensity histogram with the threshold.

    Returns:
        tuple: (binary_volume, threshold_value)
    """
    threshold = threshold_otsu(volume)

    if plot_hist:
        import matplotlib.pyplot as plt

        histogram_intensity, _ = np.histogram(volume.ravel(), bins=256, range=(0, 256))

        plt.figure(figsize=(10, 6))
        plt.title("Histograma do Volume 3D")
        plt.xlabel("Nível de Cinza (Intensidade do Voxel)")
        plt.ylabel("Frequência")
        plt.plot(histogram_intensity, color="gray")
        plt.axvline(x=threshold, color="r", linestyle="--")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    return volume > threshold, threshold
