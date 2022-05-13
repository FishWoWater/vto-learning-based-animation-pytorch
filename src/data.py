import os, os.path as osp, glob
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("/data5/hadoop/vto-learning-based-animation-pytorch")
from src.utils import load_obj, laplacianMatrix
from src.smpl import SMPLBatchModel, SMPLTorchModel

# it's actually better to preserse one-one mapping instead of using many-one mapping strategy
NUM_FRAMES_END = 1
TRAIN_RATIO = 0.8
SMOOTH_ITERS = 1

class VTOShapeDataset(Dataset):
    def __init__(self, in_root, template_path, is_train=True):
        super(VTOShapeDataset, self).__init__()
        cloth_template, cloth_faces = load_obj(template_path)
        cloth_lapmat = laplacianMatrix(cloth_faces)
        # get all the pickle files
        all_shape_files = glob.glob(osp.join(in_root, "*.pkl"))
        train_cnt = int(len(all_shape_files) * TRAIN_RATIO)
        shape_files = all_shape_files[:train_cnt] if is_train else all_shape_files[train_cnt:]
        all_betas = []; all_vertices = []
        for shape_file in shape_files:
            dat = pkl.load(open(shape_file, "rb"))
            betas, vertices, faces = dat['beta'], dat['vertices'], dat['faces']
            if vertices.ndim == 2:   vertices = vertices[np.newaxis]
            if betas.ndim == 1:  betas = betas[np.newaxis]
            vertices = vertices[-NUM_FRAMES_END:]
            betas = np.tile(betas, (NUM_FRAMES_END, 1))
            all_vertices.append(vertices)
            all_betas.append(betas)

        self.all_betas = np.concatenate(all_betas, axis=0)
        self.all_vertices = np.concatenate(all_vertices, axis=0)
        self.all_fit_disp = self.all_vertices - cloth_template[np.newaxis]
        for _ in range(SMOOTH_ITERS):
            self.all_fit_disp = np.einsum('bd,adc->abc', np.asarray(cloth_lapmat.todense()), self.all_fit_disp)

        self.all_betas = torch.from_numpy(self.all_betas).float()
        self.all_fit_disp = torch.from_numpy(self.all_fit_disp).float()

    def __len__(self):
        return len(self.all_fit_disp)

    def __getitem__(self, index):
        beta, fit_disp = self.all_betas[index], self.all_fit_disp[index]
        return beta, fit_disp

def pose_dataset_collate_fn(batch):
    num_frames_all = torch.Tensor([item['num_frames'] for item in batch]).int()
    max_seqlen = num_frames_all.max()

    # don't need to pad
    padded_shapes = np.array([np.concatenate(
        (item['shape'], np.tile(item['shape'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1))), axis=0) for
                    itemid, item in enumerate(batch)])
    padded_shapes = torch.Tensor(padded_shapes)

    padded_poses = np.array([np.concatenate(
        (item['pose'], np.tile(item['pose'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1))), axis=0) for
                     itemid, item in enumerate(batch)])
    padded_poses = torch.Tensor(padded_poses)

    padded_trans = np.array([np.concatenate(
        (item['translation'], np.tile(item['translation'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1))), axis=0) for
                     itemid, item in enumerate(batch)])
    padded_trans = torch.Tensor(padded_trans)

    padded_glb_transforms = np.array([np.concatenate(
        (item['glb_transform'], np.tile(item['glb_transform'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1, 1, 1))), axis=0) for
        itemid, item in enumerate(batch)])
    padded_glb_transforms = torch.Tensor(padded_glb_transforms)

    padded_vertices = [np.concatenate((item['vertices'], np.tile(item['vertices'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1, 1))), axis=0) for itemid, item in enumerate(batch)]
    padded_vertices = torch.Tensor(np.stack(padded_vertices, axis=0))
    return num_frames_all, padded_shapes, padded_poses, padded_glb_transforms, padded_trans, padded_vertices

class SimpleVTOPoseDataset(Dataset):
    def __init__(self, in_root, is_train=True):
        super(SimpleVTOPoseDataset, self).__init__()
        # get all the pickle files
        self.is_train = is_train
        all_seq_files = glob.glob(osp.join(in_root, "*.pkl"))
        train_cnt = int(len(all_seq_files) * TRAIN_RATIO)
        seq_files = all_seq_files[:train_cnt] if is_train else all_seq_files[train_cnt:]

        sequences = []
        print('find {} sequences'.format(len(seq_files)))
        glb_transform_dict = self._get_global_transform(seq_files)
        for seq_file in seq_files:
            dat = pkl.load(open(seq_file, "rb"))
            shape, pose, translation = dat['shape'], dat['pose'], dat['translation']
            vertices, faces, num_frames = dat['vertices'], dat['faces'], dat['num_frames']
            sequence = {
                "num_frames": num_frames,
                "vertices": vertices.astype(np.float32),
                "shape": shape.astype(np.float32),
                "pose": pose.astype(np.float32),
                "translation": translation.astype(np.float32),
                "glb_transform": glb_transform_dict[seq_file]
            }

    def __len__(self, x):
        pass

    def __getitem__(self, index):
        pass

class VTOPoseDataset(Dataset):
    def __init__(self, in_root, is_train=True, loose_only=False):
        super(VTOPoseDataset, self).__init__()
        # get all the pickle files
        self.is_train = is_train
        all_seq_files = glob.glob(osp.join(in_root, "*.pkl"))
        train_cnt = int(len(all_seq_files) * TRAIN_RATIO)
        seq_files = all_seq_files[:train_cnt] if is_train else all_seq_files[train_cnt:]

        sequences = []
        glb_transform_dict = self._get_global_transform(seq_files)
        for seq_file in seq_files:
            dat = pkl.load(open(seq_file, "rb"))
            shape, pose, translation = dat['shape'], dat['pose'], dat['translation']
            if loose_only and shape[0][1] != -2:    continue
            vertices, faces, num_frames = dat['vertices'], dat['faces'], dat['num_frames']
            sequence = {
                "num_frames": num_frames, 
                "vertices": vertices.astype(np.float32), 
                "shape": shape.astype(np.float32), 
                "pose": pose.astype(np.float32), 
                "translation": translation.astype(np.float32),
                "glb_transform": glb_transform_dict[seq_file]
            }

            sequences.append(sequence)
        print("find {} sequences".format(len(sequences)))
        self.sequences = sequences

    def _get_global_transform(self, seq_files):
        # pre-compute the global transform to speed up
        glb_transform_savepath = "assets/glb_transform_{}.pkl".format("train" if self.is_train else "val")
        if osp.exists(glb_transform_savepath):
            return pkl.load(open(glb_transform_savepath, "rb"))
        glb_transform_dict = {}
        # smpl_model = SMPLBatchModel("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
        smpl_model = SMPLTorchModel(model_path="assets/SMPL/SMPL_FEMALE.pkl")
        for seq_file in seq_files:
            print('get transform for file', seq_file)
            dat = pkl.load(open(seq_file, "rb"))
            shape, pose = dat['shape'], dat['pose']
            # smpl_model.set_params(beta=shape, pose=pose)
            # smpl_model.update()
            trans = torch.zeros((shape.shape[0], 3)).cuda()
            smpl_model.forward(torch.from_numpy(shape).type(torch.float64).cuda(), torch.from_numpy(pose).type(torch.float64).cuda(), trans)
            glb_transform_dict[seq_file] = smpl_model.global_joint_transforms.cpu().numpy()
        pkl.dump(glb_transform_dict, open(glb_transform_savepath, "wb"))
        return glb_transform_dict

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        return sequence

if __name__ == "__main__":
    def collate_fn(batch):
        num_frames_all = torch.Tensor([item['num_frames'] for item in batch]).int()
        max_seqlen = num_frames_all.max()

        # don't need to pad
        padded_shapes = np.array([np.concatenate(
            (item['shape'], np.tile(item['shape'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1))), axis=0) for
                        itemid, item in enumerate(batch)])
        padded_shapes = torch.Tensor(padded_shapes)

        padded_poses = np.array([np.concatenate(
            (item['pose'], np.tile(item['pose'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1))), axis=0) for
                         itemid, item in enumerate(batch)])
        padded_poses = torch.Tensor(padded_poses)

        padded_trans = np.array([np.concatenate(
            (item['translation'], np.tile(item['translation'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1))), axis=0) for
                         itemid, item in enumerate(batch)])
        padded_trans = torch.Tensor(padded_trans)

        padded_vertices = [np.concatenate((item['vertices'], np.tile(item['vertices'][-1:], (max_seqlen - num_frames_all[itemid].item(), 1, 1))), axis=0) for itemid, item in enumerate(batch)]
        padded_vertices = torch.Tensor(np.stack(padded_vertices, axis=0))
        return num_frames_all, padded_shapes, padded_poses, padded_vertices

    
    from torch.utils.data import DataLoader 
    pose_dataset = VTOPoseDataset("./vto-dataset/tshirt-poseseqs")
    train_loader = DataLoader(pose_dataset, batch_size=32, shuffle=True, collate_fn=pose_dataset_collate_fn, num_workers=1)
    iterator = iter(train_loader)
    a, b, c, d, e, f = next(iterator)
    print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)

