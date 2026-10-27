import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.measure import label

def label_fibers_watershed(binary_image, skeleton_3d, *, verbose=True):
    """
    Segments individual fibers using the Watershed algorithm seeded by the skeleton.
    
    The logic is:
    1. Calculate the Euclidean Distance Transform (EDT) of the binary fiber volume.
    2. Label the connected components of the skeleton to use as unique markers.
    3. Apply Watershed to expand these markers filling the fiber volume.
    
    Args:
        binary_image (np.ndarray): The binary 3D volume of the fibers (True/1 = fiber).
        skeleton_3d (np.ndarray): The processed/pruned 3D skeleton (True/1 = skeleton).
        verbose (bool): Print the number of watershed markers.
        
    Returns:
        np.ndarray: A labeled 3D volume where each fiber has a unique integer ID.
    """
    # 1. Compute Distance Transform
    # We calculate the distance from the background (0) to the center of the fibers
    distance = ndimage.distance_transform_edt(binary_image)
    
    # 2. Generate Markers from Skeleton
    # Label each connected component of the skeleton with a unique ID
    markers, num_features = label(skeleton_3d, return_num=True)
    if verbose:
        print(f"      Watershed markers: {num_features}", flush=True)
    
    # 3. Apply Watershed
    # We use -distance because watershed works on "basins" (minima), 
    # so the center of the fiber (max distance) becomes the bottom of the basin.
    # mask=binary_image ensures we only segment within the fiber regions.
    labels = watershed(-distance, markers, mask=binary_image)
    
    return labels