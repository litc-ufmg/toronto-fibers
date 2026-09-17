import numpy as np
from skimage.filters import threshold_otsu


def select_cube_region(volume, edge_length, position="center"):
    """
    Extracts a cubic region at one of three representative volume positions.

    Args:
        volume (np.ndarray): Input 3D volume with shape (Z, Y, X).
        edge_length (int): Cube edge length in voxels.
        position (str): Region to sample: ``"edge"``, ``"intermediate"``
            or ``"center"``.

    Returns:
        np.ndarray: Cubic sub-volume of shape (edge_length, edge_length, edge_length).
                    If the requested edge is larger than any axis, that axis is
                    clipped to the full extent of the volume.
    """
    position_factors = {
        "edge": 0.0,
        "intermediate": 0.25,
        "center": 0.5,
    }
    if position not in position_factors:
        valid = ", ".join(position_factors)
        raise ValueError(f"Invalid cube position '{position}'. Use one of: {valid}.")

    d, h, w = volume.shape

    if edge_length > min(d, h, w):
        print(
            f"Warning: edge_length ({edge_length}) is larger than one of the "
            f"dimensions {volume.shape}. The cube will be clipped to fit."
        )

    z_size = min(edge_length, d)
    y_size = min(edge_length, h)
    x_size = min(edge_length, w)

    factor = position_factors[position]
    z_s = round((d - z_size) * factor)
    y_s = round((h - y_size) * factor)
    x_s = round((w - x_size) * factor)

    z_e = z_s + z_size
    y_e = y_s + y_size
    x_e = x_s + x_size

    print(
        f"Cropping {position} cube: "
        f"Z[{z_s}:{z_e}], Y[{y_s}:{y_e}], X[{x_s}:{x_e}]"
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
