import os, os.path as osp
import numpy as np
import torch
torch.set_grad_enabled(False)
import open3d as o3d

from src.skinning import lbs, lbs_torch_batch
from src.utils import load_obj, save_obj, laplacianMatrix, query_closest_vertices, load_motion
from src.model import GarmentFitRegressor, GarmentWrinkleRegressor, SimpleGarmentWrinkleRegressor
from src.smpl import SMPLModel, SMPLBatchModel, SMPLTorchModel
from src.postprocess import fix_collisions

learn_shape_blend_shapes = False
device = "cuda:0"
fit_model_path = "./checkpoints/fit_model.pth.tar"
wrinkle_model_path = "./checkpoints/wrinkle_model.pth.tar.bk"
pose_path = "assets/motions/arm.npz"
save_dir = osp.join("reproduce_soc/", osp.basename(pose_path).split('.')[0])
os.makedirs(save_dir, exist_ok=True)
draped_cloth_path = "./assets/meshes/tshirt_draped.obj"
clothv, clothf = load_obj(draped_cloth_path)

# smooth the vertices
lapmat = laplacianMatrix(clothf)
for i in range(1):  clothv = np.asarray(lapmat.todense()) @ clothv
num_vertices = clothv.shape[0]

fit_model = GarmentFitRegressor(nv=num_vertices).to(device)
fit_model.load_state_dict(torch.load(fit_model_path))
print('fit model loaded')

wrinkle_model = GarmentWrinkleRegressor(nv=num_vertices).to(device)
wrinkle_model.load_state_dict(torch.load(wrinkle_model_path))
print('wrinkle model loaded')

fit_model.eval(); wrinkle_model.eval()

motion = load_motion(pose_path)
print(motion.keys())

shape, pose, trans = motion['shape'], motion['pose'], motion['translation']
shape = torch.from_numpy(shape).to(device).type(torch.float64); pose = torch.from_numpy(pose).to(device).type(torch.float64)
shape = shape[None].repeat(pose.shape[0], 1)

smpl_model = SMPLTorchModel(device, "assets/SMPL/SMPL_FEMALE.pkl")
body_verts, _ = smpl_model.forward(shape, pose)

nn = query_closest_vertices(smpl_model.v_template.cpu().numpy(), clothv)
# nn = np.loadtxt("assets/meshes/tshirt_closest_body_vertices.txt", delimiter=', ')
cloth_skin_weights = smpl_model.weights[nn]

initial_state = torch.zeros((1, 1, 1500)).to(device).float()
# disps expected to be 1 x T x V x 3
inputs = torch.cat((shape, pose[:, 3:]), dim=1)[None].float()
disps, _ = wrinkle_model(inputs, initial_state)
disps = disps[0].view(pose.shape[0], -1, 3)

# simply use shape-blend-shapes to deform
cloth_shape_blend_shapes = smpl_model.body_shape_blend_shapes[:, nn] if not learn_shape_blend_shapes else fit_model(shape)
cloth_vshaped = torch.from_numpy(clothv).to(device)[None].repeat(pose.shape[0], 1, 1).type(torch.float64) + cloth_shape_blend_shapes + smpl_model.body_pose_blend_shapes[:, nn] + disps
# extract the joint_transform matrix and cloth skin weights
cloth_vertices = lbs_torch_batch(cloth_vshaped, smpl_model.global_joint_transforms, cloth_skin_weights)
# cloth_vertices[0] = torch.from_numpy(lbs(cloth_vshaped.cpu().numpy()[0], smpl_model.global_joint_transforms[0].cpu().numpy(), cloth_skin_weights.cpu().numpy())).to(device)

# save the mesh frame by frame
for fid, (cv, bv) in enumerate(zip(cloth_vertices, body_verts)):
    cv, bv = cv.cpu().numpy(), bv.cpu().numpy()
    cv = fix_collisions(cv, bv, smpl_model.faces)
    cloth_savepath = osp.join(save_dir, "%04d_garment.obj" % fid)
    body_savepath = osp.join(save_dir, "%04d_body.obj" % fid)
    save_obj(cloth_savepath, cv, clothf)
    save_obj(body_savepath, bv, smpl_model.faces)
    print('save cloth mesh to', cloth_savepath)

    # cloth = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cv), o3d.utility.Vector3iVector(clothf))
    # o3d.visualization.draw_geometries([cloth])
