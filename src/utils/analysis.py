import numpy as np
import pandas as pd
from skan import Skeleton, summarize

def junction_mask_3d(skeleton_3d):
    """
    Creates a mask identifying junctions in a 3D skeleton.
    
    Args:
        skeleton_3d (np.ndarray): Boolean or binary 3D skeleton image.
        
    Returns:
        np.ndarray: A labeled volume where:
            0 = Background
            1 = Skeleton segments
            2 = Junction points (nodes with degree >= 3)
    """
    sk = Skeleton(skeleton_3d)
    labels = np.zeros(skeleton_3d.shape, dtype=np.uint8)

    # Label all skeleton voxels as 1 (non-junction by default)
    labels[skeleton_3d] = 1

    # Label junction voxels as 2
    j_ids = np.where(sk.degrees >= 3)[0]
    if j_ids.size:
        coords = np.round(sk.coordinates[j_ids]).astype(int)  # (z,y,x)
        z = np.clip(coords[:, 0], 0, skeleton_3d.shape[0] - 1)
        y = np.clip(coords[:, 1], 0, skeleton_3d.shape[1] - 1)
        x = np.clip(coords[:, 2], 0, skeleton_3d.shape[2] - 1)
        labels[z, y, x] = 2

    return labels

def dir_vec(junc, pt):
    """
    Calculates the normalized direction vector between a junction and a point.
    
    Args:
        junc (array-like): Coordinates of the junction point.
        pt (array-like): Coordinates of the target point.
        
    Returns:
        np.ndarray: Normalized vector pointing from junc to pt.
    """
    junc = np.asarray(junc, float)
    pt   = np.asarray(pt,   float)
    v = pt - junc
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def best_straight_pair(vectors):
    """
    Finds the pair of vectors that form the straightest line (angle closest to 180 degrees).
    
    Args:
        vectors (list of np.ndarray): List of normalized direction vectors.
        
    Returns:
        tuple: (index_1, index_2, angle_in_degrees) of the best pair.
    """
    V = np.vstack(vectors)             # (n,3) already normalized
    cos = V @ V.T                      # Cosine similarity matrix
    np.fill_diagonal(cos, +np.inf)     # Ignore diagonal (self-comparison)
    
    # Find indices of the minimum cosine value (closest to -1, which means ~180 degrees)
    i, j = np.unravel_index(np.argmin(cos), cos.shape)  
    
    # Calculate angle in degrees
    ang = np.degrees(np.arccos(np.clip(cos[i, j], -1, 1)))
    return (i, j, ang)

def calculate_branch_direction(path_coords, junction_coords, num_voxels=10):
    """
    Calculates the true trajectory of a branch by fitting a 3D line 
    to the first 'num_voxels' using Singular Value Decomposition (SVD).
    This smooths out the jagged 'staircase' effect of 3D voxels.
    """
    # Ensure the path is ordered moving AWAY from the junction
    if not np.array_equal(path_coords[0], junction_coords):
        path_coords = path_coords[::-1] # Reverse if it starts at the other end
        
    # Take a segment of the branch (up to num_voxels, but bounded by branch length)
    segment = path_coords[:num_voxels]
    
    if len(segment) < 2:
        return np.array([0.0, 0.0, 0.0])
        
    # Center the segment points around their mean
    centered = segment - np.mean(segment, axis=0)
    
    # Use SVD to find the principal axis (line of best fit)
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0] 
    
    # SVD might point the vector inwards or outwards. 
    # We force it to point AWAY from the junction.
    vec_to_end = segment[-1] - segment[0]
    if np.dot(direction, vec_to_end) < 0:
        direction = -direction
        
    # Normalize the vector
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm

def filter_skeleton(skeleton_3d_img, tam_min=5, remove_isolated=True, isolated_min=15):
    """
    Removes small noise artifacts from the skeleton.
    Filters out short spurs (Junction-to-Endpoint) and floating debris (Endpoint-to-Endpoint).
    
    Args:
        skeleton_3d_img (np.ndarray): 3D boolean skeleton image.
        tam_min (int): Minimum length (in voxels) for a spur to be kept.
        remove_isolated (bool): If True, also removes small floating skeleton pieces.
        isolated_min (int): Minimum length for an isolated branch to be kept.
        
    Returns:
        np.ndarray: Filtered skeleton image.
    """
    print("------------FILTERING THE SKELETON----------")
    _skeleton_3d = skeleton_3d_img.copy()
    skeleton_obj = Skeleton(_skeleton_3d)

    # Getting the skeleton information
    summary = summarize(skeleton_obj, separator='-')

    n_spurs_rm = 0
    n_iso_rm = 0

    # Iterate over all branches efficiently using the summary dataframe
    for branch_id, branch in summary.iterrows():
        branch_voxels = skeleton_obj.path_coordinates(branch_id)
        branch_len = len(branch_voxels)
        branch_type = branch['branch-type']

        # Type 1: SPUR (Junction-to-Endpoint)
        if branch_type == 1 and branch_len <= tam_min:
            # We must identify which end is the junction to preserve it
            src = int(branch['node-id-src'])
            deg_src = skeleton_obj.degrees[src]
            
            # If the source node is the junction, keep the first voxel [1:]
            # If the destination node is the junction, keep the last voxel [:-1]
            if deg_src >= 3:
                z, y, x = branch_voxels[1:].T
            else:
                z, y, x = branch_voxels[:-1].T
                
            _skeleton_3d[z, y, x] = False
            n_spurs_rm += 1

        # Type 0: ISOLATED DEBRIS (Endpoint-to-Endpoint)
        elif remove_isolated and branch_type == 0 and branch_len <= isolated_min:
            # Delete the entire branch (no junctions to preserve)
            z, y, x = branch_voxels.T
            _skeleton_3d[z, y, x] = False
            n_iso_rm += 1

    print(f"     - {n_spurs_rm} spurs (<= {tam_min} voxels) removed.")
    if remove_isolated:
        print(f"     - {n_iso_rm} isolated branches (<= {isolated_min} voxels) removed.")
    print("------------FILTERING ENDED---------------\n")
    
    return _skeleton_3d

def find_continuous_pairs(vectors, branches_ids, min_angle=75.0):
    """
    Iteratively finds valid pairs of branches at a junction. 
    Unlike looking for a single 'best' pair, this allows multiple 
    intersecting fibers (e.g. an X-crossing) to pass through intact.
    
    Args:
        vectors (list of np.ndarray): Direction vectors of the branches.
        branches_ids (list of int): IDs of the branches.
        min_angle (float): The minimum angle (in degrees) required to consider two branches continuous.
        
    Returns:
        tuple: (paired_branches, branches_to_prune)
    """
    paired_branches = []
    unpaired_indices = list(range(len(vectors)))
    
    # Keep finding pairs as long as we have at least 2 branches left
    while len(unpaired_indices) >= 2:
        best_ang = -1
        best_pair_idx = None
        
        # Check all combinations of currently UNPAIRED branches
        for i in range(len(unpaired_indices)):
            for j in range(i + 1, len(unpaired_indices)):
                idx1 = unpaired_indices[i]
                idx2 = unpaired_indices[j]
                
                dot_product = np.clip(np.dot(vectors[idx1], vectors[idx2]), -1.0, 1.0)
                ang = np.degrees(np.arccos(dot_product))
                
                if ang > best_ang:
                    best_ang = ang
                    best_pair_idx = (i, j) # Store their positions in the unpaired_indices list
        
        # If the best remaining pair meets the threshold, lock them in
        if best_ang >= min_angle:
            orig_idx1 = unpaired_indices[best_pair_idx[0]]
            orig_idx2 = unpaired_indices[best_pair_idx[1]]
            paired_branches.append((branches_ids[orig_idx1], branches_ids[orig_idx2]))
            
            # Remove them from the unpaired pool (pop highest index first to avoid shifting)
            unpaired_indices.pop(best_pair_idx[1])
            unpaired_indices.pop(best_pair_idx[0])
        else:
            # No remaining combinations form a valid straight-ish line
            break
            
    # Any branches left in the unpaired list are the ones we need to disconnect!
    branches_to_prune = [branches_ids[idx] for idx in unpaired_indices]
    return paired_branches, branches_to_prune


def prune_skeleton(skeleton_3d_img, local_search=True, local_range=12, min_angle=75.0, retraction_dist=4):
    """
    Prunes the skeleton by analyzing continuity at junction points. 
    It iteratively pairs branches that form continuous lines and prunes the rest.
    
    Args:
        skeleton_3d_img (np.ndarray): 3D boolean skeleton image.
        local_search (bool): If True, uses local direction vectors (PCA fitted line). 
                             If False, uses global vectors (junction to endpoint).
        local_range (int): Number of voxels to use for the PCA line fitting.
        min_angle (float): Threshold angle to accept two branches as continuous.
        retraction_dist (int): Number of voxels to delete when pruning, to pull the stub away from the junction.
        
    Returns:
        np.ndarray: Pruned skeleton image.
    """
    # Copying the original skeleton
    _skeleton_3d = skeleton_3d_img.copy()
    skeleton_obj = Skeleton(_skeleton_3d)

    # Getting the skeleton information
    summary = summarize(skeleton_obj, separator='-')

    # Getting the junction points
    junction_points_ids = np.where(skeleton_obj.degrees >= 3)[0]

    print(f"{len(junction_points_ids)} junction points found for pruning.")

    # Iterate through the junction points
    for junction_id in junction_points_ids:

        # Get the coordinates of the junction point
        junction_coords = skeleton_obj.coordinates[junction_id]

        # Get the branches connected to the junction point
        connected_branches = summary[(summary['node-id-src'] == junction_id) | (summary['node-id-dst'] == junction_id)]

        # Guarantee minimum of three branches to perform selection
        if len(connected_branches) < 3:
            continue

        vectors = []
        branches_ids = []
        
        for _, branch in connected_branches.iterrows():
            branch_id = int(branch.name)
            
            # Determine the vector for this branch
            if local_search:
                branch_voxels = skeleton_obj.path_coordinates(branch_id)
                vec = calculate_branch_direction(branch_voxels, junction_coords, num_voxels=local_range)
            else:
                other_node_id = branch['node-id-dst'] if branch['node-id-src'] == junction_id else branch['node-id-src']
                other_coords = skeleton_obj.coordinates[int(other_node_id)]
                vec = dir_vec(junction_coords, other_coords)

            vectors.append(vec)
            branches_ids.append(branch_id)
        
        # --- NEW LOGIC: Find all valid pairs dynamically ---
        paired_branches, branches_to_prune = find_continuous_pairs(vectors, branches_ids, min_angle=min_angle)

        # Disconnect ONLY the branches that failed to find a continuous partner
        # by retracting them away from the junction volume
        for branch_to_prune in branches_to_prune:
            branch_to_prune_voxels = skeleton_obj.path_coordinates(branch_to_prune)

            # Determine how many voxels to remove (ensure we don't delete the whole branch)
            dist = min(retraction_dist, len(branch_to_prune_voxels) - 2)
            dist = max(1, dist) # Always remove at least 1 voxel to break the connection

            # Slice the voxels we want to delete depending on which end is connected to the junction
            if np.array_equal(branch_to_prune_voxels[0], junction_coords):
                voxels_to_remove = branch_to_prune_voxels[1 : 1 + dist]
            else:
                if dist == 1:
                    voxels_to_remove = branch_to_prune_voxels[-2:-1]
                else:
                    voxels_to_remove = branch_to_prune_voxels[-(1 + dist) : -1]
            
            # Set all selected voxels to 0
            for voxel in voxels_to_remove:
                _skeleton_3d[tuple(voxel)] = 0
            
    return _skeleton_3d