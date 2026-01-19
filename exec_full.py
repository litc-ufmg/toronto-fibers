###########################
#  Librarys
###########################

import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from scipy.spatial import cKDTree
from skimage.color import label2rgb, gray2rgb
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt
from skimage.util import invert, img_as_float
from skimage.segmentation import watershed
from skimage.io import imread
from skimage import filters, restoration
from tqdm import tqdm
from IPython.display import clear_output
from skimage.filters import threshold_otsu
from skan import Skeleton, summarize

###########################
# Funções utilitárias
###########################

def binary_to_rgb(binary_img):
    mask = binary_img.astype(bool)
    rgb_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb_img[mask] = (255, 0, 0)
    return rgb_img

def binary_to_gray(binary_img, fg=255, bg=0):
    return np.where(binary_img, fg, bg).astype(np.uint8)

def junction_mask_3d(skeleton_3d):
    sk = Skeleton(skeleton_3d)
    labels = np.zeros(skeleton_3d.shape, dtype=np.uint8)

    # marca todos voxels do esqueleto como 1
    labels[skeleton_3d] = 1

    # marca nós de junção como 2
    j_ids = np.where(sk.degrees >= 3)[0]
    if j_ids.size:
        coords = np.round(sk.coordinates[j_ids]).astype(int)  # (z,y,x)
        z = np.clip(coords[:, 0], 0, skeleton_3d.shape[0] - 1)
        y = np.clip(coords[:, 1], 0, skeleton_3d.shape[1] - 1)
        x = np.clip(coords[:, 2], 0, skeleton_3d.shape[2] - 1)
        labels[z, y, x] = 2

    return labels

def filter_skeleton(skeleton_3d_img, tam_min):
    print("------------FILTERING THE SKELETON----------")
    print("     - Removing branches with ", tam_min, " or less voxels.")
    _skeleton_3d = skeleton_3d_img.copy()
    skeleton_obj = Skeleton(_skeleton_3d)

    summary = summarize(skeleton_obj, separator='-')
    junction_points_ids = np.where(skeleton_obj.degrees >= 3)[0]

    print("     -", len(junction_points_ids), " junction points found.")

    n_rm = 0

    for junction_id in junction_points_ids:
        connected_branches = summary[
            (summary['node-id-src'] == junction_id) | (summary['node-id-dst'] == junction_id)
        ]

        for _, branch in connected_branches.iterrows():
            branch_id = int(branch.name)
            branch_voxels = skeleton_obj.path_coordinates(branch_id)
            branch_len = len(branch_voxels)

            src = int(branch['node-id-src']); dst = int(branch['node-id-dst'])
            deg_src = int(skeleton_obj.degrees[src]); deg_dst = int(skeleton_obj.degrees[dst])
            is_jj = (deg_src >= 3 and deg_dst >= 3)
            is_je = ((deg_src >= 3 and deg_dst == 1) or (deg_dst >= 3 and deg_src == 1))

            if branch_len <= tam_min and is_je and not is_jj:
                n_rm += 1
                z, y, x = branch_voxels[1:].T  # preserva voxel do nó
                _skeleton_3d[z, y, x] = False
        
    print("     -", n_rm, " branches removed.")
    print("------------FILTERING ENDED---------------\n")
    
    return _skeleton_3d

def dir_vec(junc, pt):
    junc = np.asarray(junc, float)
    pt   = np.asarray(pt,   float)
    v = pt - junc
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def best_straight_pair(vectors):
    V = np.vstack(vectors)             # (n,3) já normalizados
    cos = V @ V.T                      # cossenos
    np.fill_diagonal(cos, +np.inf)     # ignora diagonal
    i, j = np.unravel_index(np.argmin(cos), cos.shape)  # menor cos ~ -1 => ~180°
    ang = np.degrees(np.arccos(np.clip(cos[i, j], -1, 1)))
    return (i, j, ang)

def prune_skeleton(skeleton_3d_img, local_search, local_range):
    _skeleton_3d = skeleton_3d_img.copy()
    skeleton_obj = Skeleton(_skeleton_3d)
    summary = summarize(skeleton_obj, separator='-')
    junction_points_ids = np.where(skeleton_obj.degrees >= 3)[0]

    print(len(junction_points_ids), " junction points found.")

    for junction_id in junction_points_ids:
        junction_coords = skeleton_obj.coordinates[junction_id]
        print("-> Analysing junction point {", junction_coords, "}")

        connected_branches = summary[
            (summary['node-id-src'] == junction_id) | (summary['node-id-dst'] == junction_id)
        ]

        if len(connected_branches) < 3:
            continue

        vectors = []
        branches_ids = []
        for _, branch in connected_branches.iterrows():
            if local_search:
                branch_voxels = skeleton_obj.path_coordinates(int(branch.name))
                len_branch = len(branch_voxels)

                if branch['node-id-src'] == junction_id:
                    if len_branch > local_range:
                        other_coords = branch_voxels[local_range]
                    else:
                        other_node_id = branch['node-id-dst']
                        other_coords = skeleton_obj.coordinates[int(other_node_id)]
                else:
                    if len_branch > local_range:
                        other_coords = branch_voxels[-(local_range-1)]
                    else:
                        other_node_id = branch['node-id-src']
                        other_coords = skeleton_obj.coordinates[int(other_node_id)]
            else:
                other_node_id = branch['node-id-dst'] if branch['node-id-src'] == junction_id else branch['node-id-src']
                other_coords = skeleton_obj.coordinates[int(other_node_id)]

            vectors.append(dir_vec(junction_coords, other_coords))
            branches_ids.append(int(branch.name))
        
        i, j, ang = best_straight_pair(vectors)

        print("     Branches checked: ", branches_ids)
        print("     Chosen branches: {", branches_ids[i], branches_ids[j], "} w/ angle = ", round(ang, 2))

        if ang > 90:
            branch_to_prune = int(np.delete(branches_ids, [i,j])[0])
            branch_to_prune_voxels = skeleton_obj.path_coordinates(branch_to_prune)

            if np.array_equal(branch_to_prune_voxels[0], junction_coords):
                coords_to_set_zero = tuple(branch_to_prune_voxels[1])
            else:
                coords_to_set_zero = tuple(branch_to_prune_voxels[-2])
            
            print("     Prunning Branch {", branch_to_prune, "} at coord {", coords_to_set_zero, "} \n\n")
            _skeleton_3d[coords_to_set_zero] = 0
        else:
            print("     Best angle <= 90°. Prunning all branches connected to the junction point.")
            for branch_to_prune in branches_ids:
                branch_to_prune_voxels = skeleton_obj.path_coordinates(branch_to_prune)
                if np.array_equal(branch_to_prune_voxels[0], junction_coords):
                    coords_to_set_zero = tuple(branch_to_prune_voxels[1])
                else:
                    coords_to_set_zero = tuple(branch_to_prune_voxels[-2])
                print("     Prunning Branch {", branch_to_prune, "} at coord {", coords_to_set_zero, "}")
                _skeleton_3d[coords_to_set_zero] = 0
            print("\n\n")
    
    return _skeleton_3d

def label_fibers_watershed(binary, markers):
    """
    Aplica distância euclidiana 3D e Watershed para propagar rótulos do esqueleto.
    Parâmetros:
      - binary: volume binário (bool ou 0/1)
      - markers: volume de marcadores inteiros (mesmo shape de binary)
    Retorna:
      - fibras_rotuladas: rótulos propagados pelo Watershed
    """
    binary = binary.astype(bool)
    if binary.shape != markers.shape:
        raise ValueError("binary e markers devem ter o mesmo shape.")

    distancia = distance_transform_edt(binary)
    fibras_rotuladas = watershed(-distancia, markers=markers, mask=binary)
    return fibras_rotuladas

def calculate_fiber_length(labeled_skeleton):
    regs = regionprops(labeled_skeleton)
    results = []
    for r in regs:
        n_voxels = int(r.area)
        # bbox em ordem (z0,y0,x0,z1,y1,x1)
        z0, y0, x0, z1, y1, x1 = r.bbox
        length = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2))
        results.append({"label": int(r.label),
                        "n_voxels": n_voxels,
                        "length": length})
    return pd.DataFrame(results)

def calculate_fiber_orientation(labeled_skeleton):
    regs = regionprops(labeled_skeleton)
    results = []
    for r in regs:
        if r.area < 2:
            continue
        coords_zyx = r.coords.astype(float)  # (N,3) em (z,y,x)
        if coords_zyx.shape[0] < 2:
            continue
        coords_zyx -= coords_zyx.mean(axis=0, keepdims=True)
        try:
            _, S, Vt = np.linalg.svd(coords_zyx, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[0] == 0:
            continue

        v_zyx = Vt[0]
        v_zyx /= np.linalg.norm(v_zyx)

        # mapeia para (x,y,z)
        vx, vy, vz = float(v_zyx[2]), float(v_zyx[1]), float(v_zyx[0])

        ang_x = float(np.degrees(np.arccos(np.clip(vx, -1, 1))))
        ang_y = float(np.degrees(np.arccos(np.clip(vy, -1, 1))))
        ang_z = float(np.degrees(np.arccos(np.clip(vz, -1, 1))))

        results.append({"label": int(r.label),
                        "ang_to_x": ang_x,
                        "ang_to_y": ang_y,
                        "ang_to_z": ang_z})
    return pd.DataFrame(results)

###########################
# Divisão por blocos
###########################

def _blocks_by_grid(shape, grid=(2, 2, 2), overlap=8):
    """
    Retorna lista de blocos cobrindo o volume com sobreposição.
    Cada item: {idx, core, expanded}, com fatias globais.
    """
    Z, Y, X = shape
    gz, gy, gx = grid
    z_edges = np.linspace(0, Z, gz + 1, dtype=int)
    y_edges = np.linspace(0, Y, gy + 1, dtype=int)
    x_edges = np.linspace(0, X, gx + 1, dtype=int)

    blocks = []
    for iz in range(gz):
        z0, z1 = z_edges[iz], z_edges[iz + 1]
        ez0 = max(0, z0 - overlap); ez1 = min(Z, z1 + overlap)
        for iy in range(gy):
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            ey0 = max(0, y0 - overlap); ey1 = min(Y, y1 + overlap)
            for ix in range(gx):
                x0, x1 = x_edges[ix], x_edges[ix + 1]
                ex0 = max(0, x0 - overlap); ex1 = min(X, x1 + overlap)

                blocks.append({
                    "idx": (iz, iy, ix),
                    "core":     (slice(z0, z1), slice(y0, y1), slice(x0, x1)),
                    "expanded": (slice(ez0, ez1), slice(ey0, ey1), slice(ex0, ex1)),
                })
    return blocks

def _core_mask_for_block(expanded_slices, core_slices):
    """
    Máscara booleana do core dentro do subvolume expanded.
    """
    ez0, ez1 = expanded_slices[0].start, expanded_slices[0].stop
    ey0, ey1 = expanded_slices[1].start, expanded_slices[1].stop
    ex0, ex1 = expanded_slices[2].start, expanded_slices[2].stop
    cz0, cz1 = core_slices[0].start, core_slices[0].stop
    cy0, cy1 = core_slices[1].start, core_slices[1].stop
    cx0, cx1 = core_slices[2].start, core_slices[2].stop

    z0r = cz0 - ez0; z1r = z0r + (cz1 - cz0)
    y0r = cy0 - ey0; y1r = y0r + (cy1 - cy0)
    x0r = cx0 - ex0; x1r = x0r + (cx1 - cx0)

    mask = np.zeros((ez1 - ez0, ey1 - ey0, ex1 - ex0), dtype=bool)
    if (z1r > z0r) and (y1r > y0r) and (x1r > x0r):
        mask[z0r:z1r, y0r:y1r, x0r:x1r] = True
    return mask

def process_blocks(raw_tiff_img, binary_threshold, grid=(2,2,2), overlap=8,
                   tam_min_branch=5, prune_local=False, prune_range=10):
    """
    Processa a imagem em blocos e retorna DataFrames agregados (tamanho, orientação).
    """
    ZYX = raw_tiff_img.shape
    blocks = _blocks_by_grid(ZYX, grid=grid, overlap=overlap)

    dfs_len = []
    dfs_ori = []

    for b in blocks:
        iz, iy, ix = b["idx"]
        core_s, exp_s = b["core"], b["expanded"]

        sub_raw = raw_tiff_img[exp_s]
        sub_binary = sub_raw > binary_threshold

        # Pipeline por bloco (no expanded para contexto)
        sub_skel = skeletonize(sub_binary)
        sub_skel_f = filter_skeleton(sub_skel, tam_min=tam_min_branch)
        sub_skel_p = prune_skeleton(sub_skel_f, local_search=prune_local, local_range=prune_range)

        # Mantém apenas core e relabela
        core_mask = _core_mask_for_block(exp_s, core_s)
        sub_lab = label(sub_skel_p)     # rótulos no expanded
        sub_lab[~core_mask] = 0
        sub_lab_core = label(sub_lab > 0)

        # Métricas
        df_len_b = calculate_fiber_length(sub_lab_core)
        df_ori_b = calculate_fiber_orientation(sub_lab_core)

        # Metadados de origem do bloco
        meta = {
            "block_z": iz, "block_y": iy, "block_x": ix,
            "core_z0": core_s[0].start, "core_y0": core_s[1].start, "core_x0": core_s[2].start,
            "core_z1": core_s[0].stop,  "core_y1": core_s[1].stop,  "core_x1": core_s[2].stop,
        }
        if not df_len_b.empty:
            for k, v in meta.items():
                df_len_b[k] = v
            dfs_len.append(df_len_b)
        if not df_ori_b.empty:
            for k, v in meta.items():
                df_ori_b[k] = v
            dfs_ori.append(df_ori_b)

        print(f"[Blocks] Done block {(iz,iy,ix)}; fibers: len={len(df_len_b)} ori={len(df_ori_b)}")

    df_len_all = pd.concat(dfs_len, ignore_index=True) if dfs_len else pd.DataFrame()
    df_ori_all = pd.concat(dfs_ori, ignore_index=True) if dfs_ori else pd.DataFrame()
    return df_len_all, df_ori_all

###############
# Main
###############

def main():
    # Image options:
    #   "./fiber_img/data/PPGF40.tif"
    #   "./fiber_img/data/PPGF40HNT0-1.0.tif"
    #   "./fiber_img/data/PPGF40HNT0-1.5.tif"
    #   "./fiber_img/data/PPGF40HNT0-5.tif"

    filename = "./fiber_img/data/PPGF40HNT0-1.0.tif"
    raw_tiff_img = tifffile.imread(filename)

    # Converting to 8bits gray scale
    raw_tiff_img = (raw_tiff_img / np.max(raw_tiff_img) * 255).astype(np.uint8)
    raw_tiff_img = raw_tiff_img[700:800, 700:800, 100:300]

    # Threshold (Otsu global)
    binary_threshold = threshold_otsu(raw_tiff_img)

    # Histograma só para conferência
    histogram_intensity, bins = np.histogram(raw_tiff_img.ravel(), bins=256, range=(0, 256))
    plt.figure(figsize=(10, 6))
    plt.title("Histograma do Volume 3D")
    plt.xlabel("Nível de Cinza (Intensidade do Voxel)")
    plt.ylabel("Frequência")
    plt.plot(histogram_intensity, color='gray')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=binary_threshold, color='r', linestyle='--')
    plt.show()

    # -------- Análise por blocos --------
    grid = (2, 2, 2)   # ajuste conforme desejado (gz, gy, gx)
    overlap = 8        # voxels de contexto
    df_len_blocks, df_ori_blocks = process_blocks(
        raw_tiff_img,
        binary_threshold,
        grid=grid,
        overlap=overlap,
        tam_min_branch=5,
        prune_local=False,
        prune_range=10
    )

    # Salvar resultados de blocos
    os.makedirs("./resultados", exist_ok=True)
    df_len_blocks.to_csv("./resultados/tamanho_blocos.csv", index=False)
    df_ori_blocks.to_csv("./resultados/orientacao_blocos.csv", index=False)

    print("END (blocks)")

if __name__ == "__main__":
    main()