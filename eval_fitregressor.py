#!/usr/bin/env python
# coding=utf-8

import numpy as np 
import torch 
torch.set_grad_enabled(False)
import open3d as o3d
from src.smpl import SMPLModel
from src.utils import laplacianMatrix, load_obj, save_obj
from src.model import GarmentFitRegressor
from src.postprocess import fix_collisions

draped_cloth_path = "./assets/meshes/tshirt_draped.obj"
clothv, clothf = load_obj(draped_cloth_path)
# smooth the vertices
lapmat = laplacianMatrix(clothf)
for i in range(1):
    clothv = np.asarray(lapmat.todense()) @ clothv

num_vertices = clothv.shape[0]
# helper model
smpl_model = SMPLModel("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

fit_regressor = GarmentFitRegressor(num_vertices)
fit_regressor.eval()
fit_regressor.load_state_dict(torch.load("./checkpoints/fit_model.pth.tar"))

betavals = np.load("./assets/pbns_shape_testdata.npy")
inputs = torch.from_numpy(betavals).float()

disps = fit_regressor(inputs)
print(disps.shape)

body_shapedirs = np.array(smpl_model.shapedirs)
bodyv, bodyf = np.array(smpl_model.verts), np.array(smpl_model.faces)

viser = o3d.visualization.Visualizer()
viser.create_window()
cloth_mesh = o3d.geometry.TriangleMesh()
body_mesh = o3d.geometry.TriangleMesh()
cloth_mesh.triangles = o3d.utility.Vector3iVector(clothf)
body_mesh.triangles = o3d.utility.Vector3iVector(bodyf)

disps = disps.cpu().numpy()
id = 0
for beta, disp in zip(betavals, disps):
    pre = disp + clothv
    bverts = bodyv + body_shapedirs.dot(beta)
    colors = np.zeros((pre.shape[0], 3))
    colors[..., 2] = 1.
    pre = fix_collisions(pre, bverts, bodyf)
    cloth_mesh.vertices = o3d.utility.Vector3dVector(pre)
    cloth_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    body_mesh.vertices = o3d.utility.Vector3dVector(bverts)
    if id == 0:
        viser.add_geometry(cloth_mesh)
        viser.add_geometry(body_mesh)
    else:
        viser.update_geometry(cloth_mesh)
        viser.update_geometry(body_mesh)
    viser.poll_events()
    id += 1
