import numpy as np
import torch

def lbs(vertices, joint_transforms, skinning_weights):
    T = np.tensordot(skinning_weights, joint_transforms, axes=[[1], [0]])
    vertices_homogeneous = np.hstack((vertices, np.ones([vertices.shape[0], 1])))
    vertices_posed_homogeneous = np.matmul(T, vertices_homogeneous.reshape([-1, 4, 1])).reshape([-1, 4])
    vertices_posed = vertices_posed_homogeneous[:, :3]

    return vertices_posed

def lbs_torch_batch(vertices, joint_transforms, skinning_weights):
    bs = vertices.shape[0]
    T = torch.tensordot(joint_transforms, skinning_weights, dims=[[1], [1]])
    T = T.permute(0, 3, 1, 2)
    vertices_homogeneous = torch.cat((vertices, torch.ones([bs, vertices.shape[1], 1]).type_as(vertices)), dim=2)
    vertices_posed_homogeneous = torch.matmul(T, vertices_homogeneous.reshape([bs, -1, 4, 1])).reshape([bs, -1, 4])
    vertices_posed = vertices_posed_homogeneous[:, :, :3]
    return vertices_posed
