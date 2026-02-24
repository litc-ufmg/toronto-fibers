import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
=======

>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
def plot_volume_axis(volume):
    """
    Plots a 3D image on the perspective of each axis.

    Args:
        volume (np.ndarray): 3D image
    """
    plt.figure(dpi=200)
    plt.suptitle("Slices examples")
<<<<<<< HEAD
    plt.subplot(1,3,1)
    plt.imshow(volume[volume.shape[0]/2,:,:], cmap='gray')
    plt.title("YZ axis")

    plt.subplot(1,3,2)
    plt.imshow(volume[:,volume.shape[1]/2,:], cmap='gray')
    plt.title("XZ axis")

    plt.subplot(1,3,3)
    plt.imshow(volume[:,:,volume.shape[2]/2], cmap='gray')
=======
    plt.subplot(1, 3, 1)
    plt.imshow(volume[volume.shape[0] / 2, :, :], cmap="gray")
    plt.title("YZ axis")

    plt.subplot(1, 3, 2)
    plt.imshow(volume[:, volume.shape[1] / 2, :], cmap="gray")
    plt.title("XZ axis")

    plt.subplot(1, 3, 3)
    plt.imshow(volume[:, :, volume.shape[2] / 2], cmap="gray")
>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
    plt.title("XY axis")

    plt.tight_layout()
    plt.show()


def summarize_and_plot_8bit_volume(img_8bit):
    """
    Prints details of the raw 8bit 3D image and plots it on the perspective of each axis.

    Args:
        img_8bit (np.ndarray): 8bit 3D image.
    """
<<<<<<< HEAD
    print("Dimensions raw image:",
        " \n X: ", img_8bit.shape[0], 
        " \n Y: ", img_8bit.shape[1], 
        " \n Z: ", img_8bit.shape[2])
=======
    print(
        "Dimensions raw image:",
        " \n X: ",
        img_8bit.shape[0],
        " \n Y: ",
        img_8bit.shape[1],
        " \n Z: ",
        img_8bit.shape[2],
    )
>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e

    # Axis plot
    plot_volume_axis(img_8bit)

<<<<<<< HEAD
def binary_to_rgb(binary_img):
    """
    Converts a binary image to an RGB mask (red color).
    
    Args:
        binary_img (np.ndarray): Binary image/volume.
        
=======

def binary_to_rgb(binary_img):
    """
    Converts a binary image to an RGB mask (red color).

    Args:
        binary_img (np.ndarray): Binary image/volume.

>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
    Returns:
        np.ndarray: RGB image where True/1 values are red (255, 0, 0).
    """
    mask = binary_img.astype(bool)
<<<<<<< HEAD
    rgb_img = np.zeros((*mask.shape,3), dtype=np.uint8)
    rgb_img[mask]=(255, 0, 0)
    return rgb_img

def binary_to_gray(binary_img, fg=255, bg=0):
    """
    Converts a binary image to grayscale 8-bit.
    
=======
    rgb_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb_img[mask] = (255, 0, 0)
    return rgb_img


def binary_to_gray(binary_img, fg=255, bg=0):
    """
    Converts a binary image to grayscale 8-bit.

>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
    Args:
        binary_img (np.ndarray): Binary image.
        fg (int): Intensity value for foreground (default 255).
        bg (int): Intensity value for background (default 0).
<<<<<<< HEAD
        
=======

>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
    Returns:
        np.ndarray: Grayscale uint8 image.
    """
    return np.where(binary_img, fg, bg).astype(np.uint8)

<<<<<<< HEAD
def plot_orientation_histograms(df, columns, bins=90, color="#ff7f0e"):
    """
    Plots histograms for fiber orientation angles.
    
    Extracted from the analysis cells at the end of the notebook.
    
=======

def plot_orientation_histograms(df, columns, bins=90, color="#ff7f0e"):
    """
    Plots histograms for fiber orientation angles.

    Extracted from the analysis cells at the end of the notebook.

>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
    Args:
        df (pd.DataFrame): Dataframe containing fiber properties.
        columns (list of str): List of column names to plot (e.g., ['ang_to_y', 'ang_to_z']).
        bins (int or sequence): Number of histogram bins (default 90).
        color (str): Color of the bars (default orange).
    """
    for col in columns:
        if col in df.columns:
            plt.figure(dpi=150)
            # Logic extracted from notebook snippet: plt.hist(..., bins=np.linspace(0, 180, 90)...)
<<<<<<< HEAD
            plt.hist(df[col], bins=np.linspace(0, 180, bins), color=color, edgecolor="black")
=======
            plt.hist(
                df[col], bins=np.linspace(0, 180, bins), color=color, edgecolor="black"
            )
>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
            plt.title(f"Orientation: {col}")
            plt.xlabel("Angle (degrees)")
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)
            plt.show()
        else:
<<<<<<< HEAD
            print(f"Warning: Column '{col}' not found in DataFrame.")
=======
            print(f"Warning: Column '{col}' not found in DataFrame.")
>>>>>>> be95cec6e9c5943da397724f33b67a753da3b97e
