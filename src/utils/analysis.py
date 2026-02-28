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

def filter_skeleton(skeleton_3d_img, tam_min):
    """
    Removes small branches (spurs) from the skeleton that are not connected 
    between two junction points (J-J branches are preserved).
    
    Args:
        skeleton_3d_img (np.ndarray): 3D boolean skeleton image.
        tam_min (int): Minimum length (in voxels) for a branch to be kept.
        
    Returns:
        np.ndarray: Filtered skeleton image.
    """
    print("------------FILTERING THE SKELETON----------")
    print(f"     - Removing branches with {tam_min} or less voxels.")
    
    # Copying the original skeleton to avoid modifying the input in-place
    _skeleton_3d = skeleton_3d_img.copy()
    skeleton_obj = Skeleton(_skeleton_3d)

    # Getting the skeleton information as a DataFrame
    summary = summarize(skeleton_obj, separator='-')

    # Getting the junction points (nodes with degree >= 3)
    junction_points_ids = np.where(skeleton_obj.degrees >= 3)[0]

    print(f"     - {len(junction_points_ids)} junction points found.")

    n_rm = 0

    # Iterate through the junction points
    for junction_id in junction_points_ids:

        # Get the branches connected to the current junction point
        connected_branches = summary[(summary['node-id-src'] == junction_id) | (summary['node-id-dst'] == junction_id)]

        # Iterate over connected branches
        for _, branch in connected_branches.iterrows():
            branch_id = int(branch.name)
            branch_voxels = skeleton_obj.path_coordinates(branch_id)
            branch_len = len(branch_voxels)

            # Classify the extreme nodes of the branch (Source and Destination)
            src = int(branch['node-id-src'])
            dst = int(branch['node-id-dst'])
            
            deg_src = int(skeleton_obj.degrees[src])
            deg_dst = int(skeleton_obj.degrees[dst])
            
            # Check connection types:
            # is_jj: Junction-to-Junction connection
            # is_je: Junction-to-Endpoint connection (Spur)
            is_jj = (deg_src >= 3 and deg_dst >= 3)
            is_je = ((deg_src >= 3 and deg_dst == 1) or (deg_dst >= 3 and deg_src == 1))

            # Remove only short Junction-to-Endpoint (spur) branches
            if branch_len <= tam_min and is_je and not is_jj:
                n_rm += 1
                # Set voxels to False (0), removing them from skeleton
                # We skip the first voxel [1:] to preserve the junction node itself
                z, y, x = branch_voxels[1:].T  
                _skeleton_3d[z, y, x] = False
        
    print(f"     - {n_rm} branches removed.")
    print("------------FILTERING ENDED---------------\n")
    
    return _skeleton_3d

def prune_skeleton(skeleton_3d_img, local_search=True, local_range=5):
    """
    Prunes the skeleton by analyzing continuity at junction points. 
    It keeps the pair of branches that form the straightest line and removes the others.
    
    Args:
        skeleton_3d_img (np.ndarray): 3D boolean skeleton image.
        local_search (bool): If True, uses local direction vectors (near the junction). 
                             If False, uses global vectors (junction to endpoint).
        local_range (int): Distance in voxels to calculate the local vector.
        
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
            
            # Determine the vector for this branch
            if local_search:
                branch_voxels = skeleton_obj.path_coordinates(int(branch.name))
                len_branch = len(branch_voxels)

                # Localize the start/end of the branch relative to the junction point
                if branch['node-id-src'] == junction_id:  
                    # Junction is the start of the branch
                    if len_branch > local_range:
                        other_coords = branch_voxels[local_range]
                    else:
                        other_node_id = branch['node-id-dst']
                        other_coords = skeleton_obj.coordinates[int(other_node_id)]
                else: 
                    # Junction is the end of the branch
                    if len_branch > local_range:
                        other_coords = branch_voxels[-(local_range-1)]
                    else:
                        other_node_id = branch['node-id-src']
                        other_coords = skeleton_obj.coordinates[int(other_node_id)]
            else:
                # Global search: use the other endpoint of the branch
                other_node_id = branch['node-id-dst'] if branch['node-id-src'] == junction_id else branch['node-id-src']
                other_coords = skeleton_obj.coordinates[int(other_node_id)]

            vectors.append(dir_vec(junction_coords, other_coords))
            branches_ids.append(int(branch.name))
        
        # Find the best pair of branches (closest to straight line)
        i, j, ang = best_straight_pair(vectors)

        # If the best angle is valid (e.g., > 90 degrees implies some continuity, 
        # though ideally close to 180), we prune the others.
        if ang > 90:
            # Identify the branch to prune (the one NOT in the best pair)
            # Note: This logic assumes exactly 3 branches. If >3, might need adjustment.
            # Here we select the first one that is not i or j.
            branch_to_prune = int(np.delete(branches_ids, [i, j])[0])

            branch_to_prune_voxels = skeleton_obj.path_coordinates(branch_to_prune)

            # Determine which voxel to set to zero to disconnect the branch
            if np.array_equal(branch_to_prune_voxels[0], junction_coords):
                coords_to_set_zero = tuple(branch_to_prune_voxels[1])
            else:
                coords_to_set_zero = tuple(branch_to_prune_voxels[-2])
            
            # Disconnect the branch
            _skeleton_3d[coords_to_set_zero] = 0

        else:
            # If no pair forms a good line (all angles <= 90), prune ALL branches
            # connected to this junction (it's likely a noise artifacts or knot)
            for branch_to_prune in branches_ids:
                branch_to_prune_voxels = skeleton_obj.path_coordinates(branch_to_prune)

                if np.array_equal(branch_to_prune_voxels[0], junction_coords):
                    coords_to_set_zero = tuple(branch_to_prune_voxels[1])
                else:
                    coords_to_set_zero = tuple(branch_to_prune_voxels[-2])
            
                _skeleton_3d[coords_to_set_zero] = 0
    
    return _skeleton_3d