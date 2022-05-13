import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
from src.skinning import lbs
from src.data import VTOShapeDataset, VTOPoseDataset, SimpleVTOPoseDataset, pose_dataset_collate_fn
from src.utils import load_obj, save_obj, laplacianMatrix, query_closest_vertices
from src.model import GarmentFitRegressor, GarmentWrinkleRegressor, SimpleGarmentWrinkleRegressor
from src.smpl import SMPLModel, SMPLBatchModel, SMPLTorchModel

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=2000)
parser.add_argument('--cloth_path', type=str, default='assets/meshes/tshirt.obj')
parser.add_argument('--fit_model_path', default='')
parser.add_argument('--wrinkle_model_path', default='')
parser.add_argument('--use_simple_wrinkle_regressor', action='store_true')
parser.add_argument('--exp_id', default='default')
parser.add_argument('--learn_post_disp', action='store_true')
parser.add_argument('--loose_only', action='store_true')
args = parser.parse_args()

checkpoint_dir = osp.join("checkpoints", args.exp_id)
os.makedirs(checkpoint_dir, exist_ok=True)
device = "cuda:0"
vis_interval = 50
vis_savedir = "vis/"
os.makedirs(vis_savedir, exist_ok=True)
draped_cloth_path = args.cloth_path.replace('.obj', '_draped.obj')
clothv, clothf = load_obj(draped_cloth_path)
# smooth the vertices
lapmat = laplacianMatrix(clothf)
for i in range(1):
    # clothv = np.einsum('bb,bc->bc', np.asarray(lapmat.todense()), clothv)
    clothv = np.asarray(lapmat.todense()) @ clothv

num_vertices = clothv.shape[0]
# helper model
smpl_model = SMPLModel("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

train_fit_dataset = VTOShapeDataset("vto-dataset/tshirt-shapeseqs", draped_cloth_path, is_train=True)
val_fit_dataset = VTOShapeDataset("vto-dataset/tshirt-shapeseqs", draped_cloth_path, is_train=False)

train_fit_loader = DataLoader(train_fit_dataset, batch_size=args.bs, shuffle=True)
val_fit_loader = DataLoader(val_fit_dataset, batch_size=args.bs, shuffle=False)

fit_model = GarmentFitRegressor(num_vertices).to(device)
if args.use_simple_wrinkle_regressor:   wrinkle_model = SimpleGarmentWrinkleRegressor(num_vertices).to(device)
else:   wrinkle_model = GarmentWrinkleRegressor(num_vertices).to(device)

optimizer = torch.optim.Adam(fit_model.parameters(), lr=args.lr)
optimizer.add_param_group({"params": wrinkle_model.parameters(), "lr": args.lr})

criterion = nn.MSELoss().to(device)

if args.fit_model_path:
    fit_model.load_state_dict(torch.load(args.fit_model_path))
else:
    print('--- training the fit regressor ---')
    for epoch in range(args.num_epochs):
        fit_model.train()
        for iter, (beta, fit_disp) in enumerate(train_fit_loader):
            beta, fit_disp = beta.to(device), fit_disp.to(device)
            optimizer.zero_grad()
            pred_fit_disp = fit_model(beta)
            loss = criterion(pred_fit_disp, fit_disp)
            loss.backward()
            optimizer.step()

        fit_model.eval()
        total_loss_val = 0.
        with torch.no_grad():
            for iter, (beta, fit_disp) in enumerate(val_fit_loader):
                beta, fit_disp = beta.to(device), fit_disp.to(device)
                pred_fit_disp = fit_model(beta)
                loss = criterion(pred_fit_disp, fit_disp)

                total_loss_val = total_loss_val + loss

                # zeropose = np.zeros((72,))
                # zerotrans = np.zeros((3,))
                # smpl_model.set_params(pose=zeropose, beta=beta.cpu().numpy()[0], trans=zerotrans)
                # smpl_model.update()
                # body_path = osp.join("visgt", "{}_{}_body.obj".format(epoch, iter))
                # cloth_path = osp.join("visgt", "{}_{}_garment.obj".format(epoch, iter))
                # savepath = osp.join("visgt", "{}_{}.png".format(epoch, iter))
                # save_obj(body_path, smpl_model.verts, smpl_model.faces)
                # save_obj(cloth_path, fit_disp[0].cpu().numpy() + clothv, clothf)

            betas_to_be_tested = np.zeros((20, 10))
            for i in range(4):
                betas_to_be_tested[i*5:(i+1)*5, i] = np.linspace(-2., 2., 5)

            if epoch % vis_interval == 0:
                for betaid, beta in enumerate(betas_to_be_tested):
                    beta = torch.from_numpy(beta.reshape(1, -1)).float().to(device)
                    pred_fit_disp = fit_model(beta)
                    zeropose = np.zeros((72,))
                    zerotrans = np.zeros((3,))
                    smpl_model.set_params(pose=zeropose, beta=beta.cpu().numpy()[0], trans=zerotrans)
                    smpl_model.update()
                    body_path = osp.join(vis_savedir, "{}_{}_body.obj".format(epoch, betaid))
                    cloth_path = osp.join(vis_savedir, "{}_{}_garment.obj".format(epoch, betaid))
                    savepath = osp.join(vis_savedir, "{}_{}.png".format(epoch, betaid))
                    save_obj(body_path, smpl_model.verts, smpl_model.faces)
                    save_obj(cloth_path, pred_fit_disp[0].cpu().numpy() + clothv, clothf)
                    render_cmd = "blender --background rendering/scene.blend --python rendering/render_helper.py --cloth_path {} --body_path {} --save_path {}".format(cloth_path, body_path, savepath)
                    # os.system(render_cmd)

            total_loss_val /= len(val_fit_loader)

        print('epoch: {}, validation loss: {}'.format(epoch, total_loss_val.item()))

        checkpoint_savepath = osp.join(checkpoint_dir, "fit_model.pth.tar")
        torch.save(fit_model.state_dict(), checkpoint_savepath)

print('---- training the wrinkle model ----')
bs = 8
dataset_func = SimpleVTOPoseDataset if args.use_simple_wrinkle_regressor else VTOPoseDataset
train_wrinkle_dataset = dataset_func("vto-dataset/tshirt-poseseqs", is_train=True, loose_only=args.loose_only)
val_wrinkle_dataset = dataset_func("vto-dataset/tshirt-poseseqs", is_train=False, loose_only=args.loose_only)

train_wrinkle_loader = DataLoader(train_wrinkle_dataset, batch_size=bs, shuffle=True, collate_fn=pose_dataset_collate_fn)
val_wrinkle_loader = DataLoader(val_wrinkle_dataset, batch_size=bs, shuffle=False, collate_fn=pose_dataset_collate_fn)

smpl_model = SMPLTorchModel(device, "assets/SMPL/SMPL_FEMALE.pkl")
clothv = torch.from_numpy(clothv).type(torch.float64).to(device)
closest_indices = query_closest_vertices(smpl_model.v_template.cpu().numpy(), clothv.cpu().numpy())
print(closest_indices)
cloth_skinweights = smpl_model.weights[closest_indices]
print(cloth_skinweights)
exit(0)
cloth_shapedirs = smpl_model.shapedirs[closest_indices]
cloth_posedirs = smpl_model.posedirs[closest_indices]
use_tbptt = False
tbptt_steps = 90

def chunk_splits(seqlen, shapes, poses, vts):
    max_seqlen = seqlen.max().item()
    num_chunks = max_seqlen // tbptt_steps
    if max_seqlen % tbptt_steps:    num_chunks += 1
    seqlen = [[seqlen[j].item() % tbptt_steps if seqlen[j] >= i * tbptt_steps else 0 for j in range(len(seqlen))] for i in range(num_chunks)]
    shapes = shapes.split(tbptt_steps, dim=1)
    poses = poses.split(tbptt_steps, dim=1)
    vts = vts.split(tbptt_steps, dim=1)
    return seqlen, shapes, poses, vts

for epoch in range(args.num_epochs):
    wrinkle_model.train()
    total_train_loss = 0.; total_train_loss_ref = 0
    train_cnt = 0
    if args.use_simple_wrinkle_regressor:
        pass
    else:
        for seqlen, shapes, poses, glb_transforms, trans, vts in train_wrinkle_loader:
            shapes = shapes.to(device); poses = poses.to(device)
            vts = vts.to(device); seqlen = seqlen.to(device); trans = trans.to(device)
            glb_transforms = glb_transforms.to(device); bs = shapes.shape[0]
            if not use_tbptt:
                optimizer.zero_grad()
                # data preparation and network forward pass
                inputs = torch.cat((shapes, poses[..., 3:]), dim=-1)
                pack_inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, batch_first=True, lengths=seqlen.cpu(),
                                                                      enforce_sorted=False)
                # num layers: 1, hidden size: 1500
                initial_state = torch.zeros((1, 32, 1500)).to(device)
                # disps' shape N x T x F
                disps, hidden_states = wrinkle_model(pack_inputs, initial_state)
                # disps = disps.view(disps.size(0), disps.size(1), -1, 3)

                #### obtain the cloth pose-blend-shapes
                shapes = shapes.view(-1, 10); poses = poses.view(-1, 72);  body_transform = glb_transforms.view(-1, 24, 4, 4)
                bt = body_transform.shape[0]
                # simply use shape-blend-shapes
                cloth_vshaped = torch.tensordot(shapes.type(torch.float64), cloth_shapedirs, dims=([1], [2])) + clothv[None]
                # get the pose-blend-shapes
                R_cube_big = smpl_model.rodrigues(poses.view(-1, 1, 3)).reshape(poses.shape[0], -1, 3, 3)
                R_cube = R_cube_big[:, 1:, :, :]
                I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
                          torch.zeros((poses.shape[0], R_cube.shape[1], 3, 3), dtype=torch.float64)).to(device)
                lrotmin = (R_cube - I_cube).reshape(poses.shape[0], -1, 1).squeeze(dim=2)
                cloth_pose_blend_shapes = torch.tensordot(lrotmin, cloth_posedirs, dims=([1], [2]))
                #### aggregate all factors
                cloth_vshaped_off = cloth_vshaped + cloth_pose_blend_shapes
                if not args.learn_post_disp:    cloth_vshaped_off = cloth_vshaped_off + disps.view(bt, -1, 3)

                ### cloth lbs process
                cloth_transform = torch.tensordot(cloth_skinweights, body_transform.type(torch.float64), dims=[[1], [1]])
                # cloth_vertices = lbs(clothv + )
                # => V x (B x F) x 4 x 4
                cloth_transform = cloth_transform.permute(1, 0, 2, 3)
                rest_shape_h = torch.cat(
                  (cloth_vshaped_off, torch.ones((bt, cloth_vshaped_off.shape[1], 1), dtype=torch.float64).to(device)), dim=2
                )
                cloth_vposed = torch.matmul(cloth_transform, torch.reshape(rest_shape_h, (bt, -1, 4, 1)))
                cloth_vposed = torch.reshape(cloth_vposed, (bt, -1, 4))[:, :, :3].contiguous()
                if args.learn_post_disp:    cloth_vshaped = cloth_vposed + disps.view(bt, -1, 3)
                cloth_vposed = cloth_vposed.view(bs, -1, cloth_vposed.size(1), cloth_vposed.size(2))

                rest_shape_h_lbs = torch.cat(
                    (cloth_vshaped, torch.ones((bt, cloth_vshaped.shape[1], 1), dtype=torch.float64).to(device)),
                    dim=2
                )
                cloth_vposed_lbs = torch.matmul(cloth_transform, torch.reshape(rest_shape_h_lbs, (bt, -1, 4, 1)))
                cloth_vposed_lbs = torch.reshape(cloth_vposed_lbs, (bt, -1, 4))[:, :, :3].contiguous()
                cloth_vposed_lbs = cloth_vposed_lbs.view(bs, -1, cloth_vposed_lbs.size(1), cloth_vposed_lbs.size(2))

                loss = 0.; lossref = 0.
                # targets = (vts - cloth_vposed).type(torch.float32)
                for sample_id, sample_seqlen in enumerate(seqlen):
                    # sample_loss = criterion(disps[sample_id, :sample_seqlen], targets[sample_id, :sample_seqlen])
                    offset = trans[sample_id][:sample_seqlen][:, None]
                    # import ipdb; ipdb.set_trace()
                    # sample_loss = criterion(vts[sample_id, :sample_seqlen], cloth_vposed[sample_id, :sample_seqlen].type(torch.float32))
                    sample_loss = criterion(vts[sample_id, :sample_seqlen] - offset,
                                            cloth_vposed[sample_id, :sample_seqlen].type(torch.float32))
                    sample_loss_lbsref = criterion(vts[sample_id, :sample_seqlen] - offset,
                                            cloth_vposed_lbs[sample_id, :sample_seqlen].type(torch.float32))
                    loss = loss + sample_loss
                    lossref = lossref + sample_loss_lbsref
                loss = loss / len(seqlen)
                lossref = lossref / len(seqlen)
                # print("train_cnt", train_cnt, " ", loss.item())
                total_train_loss = total_train_loss + loss.item()
                total_train_loss_ref = total_train_loss_ref + lossref.item()
                loss.backward()
                optimizer.step()
                train_cnt += 1
                if epoch % vis_interval == 0 and train_cnt == 1:
                    sid = np.random.randint(0, len(seqlen))
                    sample_seqlen = seqlen[sid]
                    body_shapes, body_poses = shapes.reshape(bs, -1, 10)[sid, :sample_seqlen], poses.reshape(bs, -1, 72)[sid, :sample_seqlen]
                    body_trans = torch.zeros((sample_seqlen, 3)).to(device)
                    body_vertices, _ = smpl_model.forward(body_shapes.type(torch.float64), body_poses.type(torch.float64), body_trans.type(torch.float64))
                    for frame_id in range(0, sample_seqlen, 10):
                        body_savepath = osp.join(vis_savedir, "wrinkle_{}_{}_{}_body.obj".format(epoch, train_cnt, frame_id))
                        cloth_savepath = osp.join(vis_savedir,
                                                 "wrinkle_{}_{}_{}_cloth.obj".format(epoch, train_cnt, frame_id))
                        cloth_trans = trans[sid][frame_id].cpu().numpy().reshape(-1, 3)
                        save_obj(body_savepath, body_vertices.cpu().numpy()[frame_id] + cloth_trans, smpl_model.faces)
                        save_obj(cloth_savepath.replace('.obj', '_pre.obj'), cloth_vposed[sid][frame_id].detach().cpu().numpy() + cloth_trans, clothf)
                        save_obj(cloth_savepath.replace('.obj', '_gt.obj'), vts[sid][frame_id].detach().cpu().numpy(), clothf)
                        save_obj(cloth_savepath.replace('.obj', '_lbs.obj'), cloth_vposed_lbs[sid][frame_id].detach().cpu().numpy() + cloth_trans, clothf)
                        print('debug obj save to', body_savepath, cloth_savepath)
            else:
                seqlen, shapes, poses, vts = chunk_splits(seqlen, shapes, poses, vts)
                initial_state = torch.zeros((1, 32, 1500)).to(device)
                wrinkle_model.gru_state = initial_state
                for _seqlen, _shapes, _poses, _vts in zip(seqlen, shapes, poses, vts):
                    # discard bad samples
                    valid_indices = torch.nonzero(torch.tensor(_seqlen)).squeeze().numpy().tolist()
                    _shapes = _shapes[valid_indices]; _poses = _poses[valid_indices]; _vts = _vts[valid_indices]; _seqlen = np.array(_seqlen)[valid_indices].tolist()
                    # discard too small batch
                    if len(valid_indices) < 5:  continue
                    optimizer.zero_grad()
                    inputs = torch.cat((_shapes, _poses[..., 3:]), dim=-1)
                    print(inputs.shape)
                    pack_inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, batch_first=True, lengths=_seqlen, enforce_sorted=False)
                    # num layers: 1, hidden size: 1500
                    input_state = wrinkle_model.gru_state[:, valid_indices]
                    disps, hidden_states = wrinkle_model(pack_inputs, input_state)
                    # calculate the loss value
                    optimizer.step()
                    wrinkle_model.repackage_rnn_state()
                    wrinkle_model.gru_state[:, valid_indices] = hidden_states[:, -1][None]
                    import ipdb; ipdb.set_trace()

        total_train_loss /= len(train_wrinkle_loader)
        print("epoch: {} training loss: {} training lbs reference loss: {}".format(epoch, total_train_loss, total_train_loss_ref))
        torch.save(wrinkle_model.state_dict(), osp.join(checkpoint_dir, "wrinkle_model.pth.tar"))
        wrinkle_model.eval()




