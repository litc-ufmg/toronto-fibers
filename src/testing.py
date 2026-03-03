import os
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
import tifffile
from  scipy.ndimage import binary_opening, binary_closing

# Import your custom modules
# Note: 'import io' can sometimes conflict with Python's standard 'io' library. 
# If you get an error here, rename your 'io.py' to 'io_utils.py' and update this import.
import utils.io as custom_io 
import utils.preprocessing as prep
import utils.analysis as analysis
import utils.segmentation as seg
import utils.visualization as vis

def generate_synthetic_crossing():
    """
    Generates a synthetic 3D volume with two thick intersecting fibers.
    This creates a perfect degree-4 junction to test the recent analysis.py fix.
    """
    print("Generating synthetic 3D crossing volume...")
    vol = np.zeros((50, 50, 50), dtype=np.uint8)
    
    # Fiber 1: Straight line along the X axis
    vol[23:27, 23:27, 10:40] = 255
    
    # Fiber 2: Straight line along the Y axis
    vol[23:27, 10:40, 23:27] = 255
    
    return vol

def run_pipeline(image_path=None, subcube_size=None):
    """
    Runs the full fiber segmentation pipeline.
    """
    print("========================================")
    print("       STARTING PIPELINE TEST")
    print("========================================")
    
    # 1. LOAD IMAGE
    if image_path and os.path.exists(image_path):
        print(f"[1/5] Loading image from: {image_path}")
        # Depending on your python environment, you might need to use standard tifffile 
        # directly if custom_io conflicts with python's built-in io module.
        volume = custom_io.load_microct_image(image_path)
        
        if subcube_size is not None:
            print(f"      - Extracting subvolume of size {subcube_size}^3 from the center for quick testing...")
            d, h, w = volume.shape
            
            # Calculate starting indices for a center crop
            z_start = max(0, (d - subcube_size) // 2)
            y_start = max(0, (h - subcube_size) // 2)
            x_start = max(0, (w - subcube_size) // 2)

            # z_start = max(0, subcube_size)
            # y_start = max(0, subcube_size)
            # x_start = max(0, subcube_size)
            
            # Calculate ending indices
            z_end = min(d, z_start + subcube_size)
            y_end = min(h, y_start + subcube_size)
            x_end = min(w, x_start + subcube_size)
            
            # Slice the center subcube
            volume = volume[z_start:z_end, y_start:y_end, x_start:x_end]
            
    else:
        print(f"[1/5] Image path not provided or not found. Using synthetic data.")
        volume = generate_synthetic_crossing()

    # 2. PREPROCESSING
    print("[2/5] Preprocessing volume (Cropping and Binarizing)...")
    cropped_volume = prep.auto_crop_volume(volume)
    
    # Apply Otsu thresholding to create the binary mask (common step from your notebook)
    thresh = threshold_otsu(cropped_volume)
    binary_image = cropped_volume > thresh
    print(f"      - Binarization complete. Foreground voxels: {np.sum(binary_image)}")

    # 3. SKELETONIZATION
    print("[3/5] Extracting 3D skeleton...")
    binary_image = binary_opening(binary_image)
    skeleton_3d = skeletonize(binary_image)
    print(f"      - Skeletonization complete. Skeleton voxels: {np.sum(skeleton_3d)}")

    # 4. ANALYSIS & PRUNING (The step we just fixed)
    print("[4/5] Analyzing junctions and pruning branches...")
    # Assuming prune_branches_by_angle modifies the skeleton in-place or returns it.
    # Make sure this matches your analysis.py exact signature.
    for _ in range(3):
        skeleton_3d = analysis.filter_skeleton(skeleton_3d, 5)
    pruned_skeleton = analysis.prune_skeleton(skeleton_3d)
    
    # 5. SEGMENTATION
    print("[5/5] Running Marker-Based Watershed Segmentation...")
    labeled_fibers = seg.label_fibers_watershed(binary_image, pruned_skeleton)
    
    # RESULTS
    num_fibers = len(np.unique(labeled_fibers)) - 1 # Subtract 1 to ignore background (0)
    print("========================================")
    print(f"PIPELINE FINISHED: Found {num_fibers} unique fibers.")
    print("========================================")
    
    # Optional: Quick visualization of the middle slice
    # Uncomment if you want to see the 2D cross section
    # vis.plot_volume_axis(labeled_fibers)

    return cropped_volume, binary_image, pruned_skeleton, labeled_fibers

if __name__ == "__main__":
    # To test with a real file, replace this with your file path:
    paths = [
        "data/PPGF40.tif",
        "data/PPGF40HNT0-1.0.tif",
        "data/PPGF40HNT0-1.5.tif",
        "data/PPGF40HNT0-5.tif",
    ]
    sample_path = paths[1]
    
    # Set subcube_size to an integer (e.g., 100 or 200) to test a small chunk of your .tif
    # Set to None to process the whole image.
    test_cube_size = 200
    
    cropped_vol, binary_mask, pruned_skeleton, final_labels = run_pipeline(sample_path, subcube_size=test_cube_size)
    
    # Ensure the output directory exists
    output_dir = "./results/fiber_img/3d-res"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths for all matrices
    out_labels_path = os.path.join(output_dir, "labeled_fibers_test.tif")
    out_crop_path = os.path.join(output_dir, "cropped_vol_test.tif")
    out_mask_path = os.path.join(output_dir, "binary_mask_test.tif")
    out_skel_path = os.path.join(output_dir, "pruned_skel_test.tif")
    
    # Save the matrices as TIF files
    # final_labels is saved as uint16 to support > 255 fibers
    print(f"Saving segmented fibers to {out_labels_path}...")
    tifffile.imwrite(out_labels_path, final_labels.astype(np.uint16))
    
    print(f"Saving cropped volume to {out_crop_path}...")
    tifffile.imwrite(out_crop_path, cropped_vol.astype(np.uint8))
    
    # Boolean arrays (True/False) are multiplied by 255 and cast to uint8 for easy viewing in Napari
    print(f"Saving binary mask to {out_mask_path}...")
    tifffile.imwrite(out_mask_path, (binary_mask.astype(np.uint8) * 255))
    
    print(f"Saving pruned skeleton to {out_skel_path}...")
    tifffile.imwrite(out_skel_path, (pruned_skeleton.astype(np.uint8) * 255))
    
    # Print the PowerShell commands for the user
    # Using 'r' (raw string) so Python doesn't treat \U or \H as escape characters
    print("\n=======================================================================")
    print("To visualize the results in Napari, run the following command in PowerShell:")
    print("scp homero_castro@litc59:/home/homero_castro/toronto/results/fiber_img/3d-res/labeled_fibers_test.tif C:\\Users\Homero\documents\LITC\Toronto_Fibers\data\\3d-res")
    print("napari data/3d-res/labeled_fibers_test.tif --layer-type labels")
    print("=======================================================================")
    print("To visualize the cropped volume in Napari, run the following command in PowerShell:")
    print("scp homero_castro@litc59:/home/homero_castro/toronto/results/fiber_img/3d-res/cropped_vol_test.tif C:\\Users\Homero\documents\LITC\Toronto_Fibers\data\\3d-res")
    print("napari data/3d-res/cropped_vol_test.tif")
    print("=======================================================================")
    print("To visualize the binary image in Napari, run the following command in PowerShell:")
    print("scp homero_castro@litc59:/home/homero_castro/toronto/results/fiber_img/3d-res/binary_mask_test.tif C:\\Users\Homero\documents\LITC\Toronto_Fibers\data\\3d-res")
    print("napari data/3d-res/binary_mask_test.tif --layer-type labels")
    print("=======================================================================")
    print("To visualize the prunned skeleton in Napari, run the following command in PowerShell:")
    print("scp homero_castro@litc59:/home/homero_castro/toronto/results/fiber_img/3d-res/pruned_skel_test.tif C:\\Users\Homero\documents\LITC\Toronto_Fibers\data\\3d-res")
    print("napari data/3d-res/pruned_skel_test.tif")
    print("=======================================================================\n")