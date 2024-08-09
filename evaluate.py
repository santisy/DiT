import argparse
import os
import glob
import math
import random

import torch
from utils.chamfer_dist import chamfer_3DDist
import trimesh
from pysdf import SDF
import numpy as np


def normalize_mesh(mesh: trimesh.Geometry) -> trimesh.Geometry:
    # Calculate the bounding box
    bbox = mesh.bounds
    # Compute the center
    center = (bbox[0] + bbox[1]) / 2.0
    mesh.apply_translation(-center)
    # Calculate scales in x, y, z directions
    x_scale = bbox[1][0] - bbox[0][0]
    y_scale = bbox[1][1] - bbox[0][1]
    z_scale = bbox[1][2] - bbox[0][2]
    # Compute the maximum diagonal
    max_diagonal = np.sqrt(x_scale**2 + y_scale**2 + z_scale**2)
    # Normalize the mesh by the diagonal
    mesh.apply_scale(1.0 / max_diagonal)
    return mesh

def measure_metrics(args):
    random.seed(0)

    # Names
    gen_name = os.path.basename(args.gen_dir)

    # Prepare to write to text file
    f = open(f"./metric_out/{gen_name}_metric_out.txt", "w")

    gen_mesh_list = sorted(glob.glob(os.path.join(args.gen_dir, "*.obj")))
    gen_points = []
    ref_mesh_list = sorted(glob.glob(os.path.join(args.ref_dir, "*.obj")))
    sample_len = int(math.ceil(len(ref_mesh_list) * 0.05))
    ref_mesh_list = random.sample(ref_mesh_list, sample_len)
    ref_points = []

    # Distance preparation
    chamfer_dist = chamfer_3DDist()

    # Sample points on generated meshes
    for obj_path in gen_mesh_list:
        mesh = trimesh.load(obj_path)
        mesh = normalize_mesh(mesh)
        f = SDF(mesh.vertices, mesh.faces)
        # This is the numpy array of points of shape (5000, 3)
        gen_points.append(f.sample_surface(5000))
    for obj_path in ref_mesh_list:
        mesh = trimesh.load(obj_path)
        mesh = normalize_mesh(mesh)
        f = SDF(mesh.vertices, mesh.faces)
        # This is the numpy array of points of shape (5000, 3)
        ref_points.append(f.sample_surface(5000))

    # Formulate to torch tensor
    gen_points_tensor = torch.from_numpy(np.stack(gen_points, axis=0)).cuda()
    ref_points_tensor = torch.from_numpy(np.stack(ref_points, axis=0)).cuda()
    total_points = torch.cat([gen_points_tensor, ref_points_tensor], dim=0)
    gen_n = gen_points_tensor.shape[0]
    ref_n = ref_points_tensor.shape[0]
    assert gen_n == ref_n

    # CD-related Metrics preparation accumulation
    min_dist = 0
    cov_set = set()
    nna_indicator = 0

    for i in range(gen_n):
        gen_points_now = gen_points_tensor[i].unsqueeze(dim=0).repeat((total_points.shape[0] - 1, 1, 1))
        rest_points = torch.cat([total_points[:i], total_points[i+1:]], dim=0).contiguous()
        dist1, dist2, _, _ = chamfer_dist(gen_points_now, rest_points)
        dist = dist1.sum(dim=1) + dist2.sum(dim=1)

        # COV
        cov_set.add(torch.argmin(dist[gen_n-1:], dim=0).item())
        # MMD
        min_dist += torch.min(dist[gen_n-1:], dim=0)[0].item()
        # 1-NNA: Belong to itself
        total_min = torch.argmin(dist, dim=0).item()
        if total_min < gen_n - 1:
            nna_indicator += 1

    for i in range(ref_n):
        ref_points_now = ref_points_tensor[i].unsqueeze(dim=0).repeat((total_points.shape[0] - 1, 1, 1))
        rest_points = torch.cat([total_points[:gen_n + i], total_points[gen_n + i+1:]], dim=0).contiguous()
        dist1, dist2, _, _ = chamfer_dist(ref_points_now, rest_points)
        dist = dist1.sum(dim=1) + dist2.sum(dim=1)

        # 1-NNA: Belong to itself
        total_min = torch.argmin(dist, dim=0).item()
        if total_min > gen_n - 1:
            nna_indicator += 1

    # Write the results
    f.write("CD results:\n")
    cov_cd = len(cov_set) / float(ref_n) * 100
    f.write(f"COV: {cov_cd:.2f}")
    mmd_cd = min_dist / float(ref_n) / 1000.
    f.write(f"MMD: {mmd_cd:.2f}")
    nna_1 = total_min / float(gen_n + ref_n) * 100
    f.write(f"1-NNA: {nna_1:.2f}") 

    # Finalize
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref_dir", type=str, required=True)
    parser.add_argument("-g", "--gen_dir", type=str, required=True)

    args = parser.parse_args()
    measure_metrics(args)
