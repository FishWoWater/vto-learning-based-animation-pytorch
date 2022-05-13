import numpy as np
import pickle
import torch
from torch.nn import Module

# to support batch inference
class SMPLBatchModel():
  def __init__(self, model_path):
    with open(model_path, 'rb') as f:
      params = pickle.load(f, encoding="latin1")

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None
    self.global_joint_transforms = None
    self.pose_blendshape = None
    self.shape_blendshape = None

    # self.update()

  def set_params(self, pose=None, beta=None, trans=None):
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if trans is not None:
      self.trans = trans
    self.update()
    return self.verts

  def update(self):
    batch_num = self.beta.shape[0]
    # how beta affect body shape
    # import ipdb; ipdb.set_trace()
    v_shaped = self.v_template + np.tensordot(self.beta, self.shapedirs, axes=[[1], [2]])
    # joints location
    # self.J = np.matmul(self.J_regressor, v_shaped)
    self.J = np.einsum('abc,db->adc', v_shaped, self.J_regressor.todense())
    # self.J = v_shaped.transpose(0, 2, 1).dot(self.J_regressor.T).transpose(0, 2, 1)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    R_cube_big = self.rodrigues(pose_cube).reshape(batch_num, -1, 3, 3)

    R_cube = R_cube_big[:, 1:]
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (batch_num, R_cube.shape[1], 3, 3)
    )
    lrotmin = (R_cube - I_cube).reshape(batch_num, -1)
    # how pose affect body shape in zero pose
    # self.pose_blendshape = self.posedirs.dot(lrotmin)
    v_posed = v_shaped + np.tensordot(lrotmin, self.posedirs, axes=[[1], [2]])
    # world transformation of each joint
    # G = np.empty((self.kintree_table.shape[1], 4, 4))
    G = []
    G.append(
        self.with_zeros(np.concatenate((R_cube_big[:, 0], self.J[:, 0, :].reshape([-1, 3, 1])), axis=2))
    )

    for i in range(1, self.kintree_table.shape[1]):
      G.append(G[self.parent[i]].dot(
        self.with_zeros(
          np.concatenate(
              [R_cube_big[:, i], ((self.J[:, i, :] - self.J[:, self.parent[i], :]).reshape([-1, 3, 1]))], axis=2
            )
        )
      ))

    G = np.stack(G, axis=1)
    # remove the transformation due to the rest pose
    G = G - self.pack(
      np.matmul(
        G,
        np.concatenate([self.J, np.zeros([batch_num, 24, 1])], axis=2).reshape([batch_num, 24, 4, 1])
      )
    )
    self.global_joint_transforms = G
    # transformation of each vertex
    # T = np.tensordot(self.weights, G, axes=[[1], [0]])
    T = np.tensordot(G, self.weights, axes=((1), (1)))
    T = np.transpose(T, (0, 3, 1, 2))
    rest_shape_h = np.concatenate((v_posed, np.ones([batch_num, v_posed.shape[1], 1])), axis=2)
    v = np.matmul(T, rest_shape_h.reshape([batch_num, -1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([-1, 1, 3])

  def rodrigues(self, r):
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, 1e-6)
    r_hat = r / theta

    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    return np.concatenate((x, np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]]), (x.shape[0], 1, 1))), axis=1)

  def pack(self, x):
    return np.concatenate((np.zeros((x.shape[0], x.shape[1], 4, 3)), x), axis=3)

  def save_to_obj(self, path):
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class SMPLModel():
  def __init__(self, model_path):
    with open(model_path, 'rb') as f:
      params = pickle.load(f, encoding="latin1")

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None
    self.global_joint_transforms = None
    self.pose_blendshape = None
    self.shape_blendshape = None

    self.update()


  def set_params(self, pose=None, beta=None, trans=None):
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if trans is not None:
      self.trans = trans
    self.update()
    return self.verts


  def update(self):
    # how beta affect body shape
    self.shape_blendshape = self.shapedirs.dot(self.beta)
    v_shaped = self.v_template + self.shape_blendshape
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()
    # how pose affect body shape in zero pose
    self.pose_blendshape = self.posedirs.dot(lrotmin)
    v_posed = v_shaped + self.pose_blendshape
    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
          )
        )
      )
    # remove the transformation due to the rest pose
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    self.global_joint_transforms = G
    # transformation of each vertex
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3])


  def rodrigues(self, r):
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, 1e-6)
    r_hat = r / theta
  
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R


  def with_zeros(self, x):
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


  def pack(self, x):
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))


  def save_to_obj(self, path):
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class SMPLTorchModel(Module):
  def __init__(self, device="cuda:0", model_path='./assets/SMPL/SMPL_FEMALE.pkl'):

    super(SMPLTorchModel, self).__init__()
    with open(model_path, 'rb') as f:
      params = pickle.load(f, encoding='latin1')
    self.J_regressor = torch.from_numpy(
      np.array(params['J_regressor'].todense())
    ).type(torch.float64)
    if 'joint_regressor' in params.keys():
      self.joint_regressor = torch.from_numpy(
        np.array(params['joint_regressor'].T.todense())
      ).type(torch.float64)
    else:
      self.joint_regressor = torch.from_numpy(
        np.array(params['J_regressor'].todense())
      ).type(torch.float64)
    self.weights = torch.from_numpy(np.array(params['weights'])).type(torch.float64)
    self.posedirs = torch.from_numpy(np.array(params['posedirs'])).type(torch.float64)
    self.v_template = torch.from_numpy(np.array(params['v_template'])).type(torch.float64)
    self.shapedirs = torch.from_numpy(np.array(params['shapedirs'])).type(torch.float64)
    self.kintree_table = np.array(params['kintree_table'])
    self.faces = np.array(params['f'])
    self.device = device if device is not None else torch.device('cpu')
    for name in ['J_regressor', 'joint_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
      _tensor = getattr(self, name)
      print(' Tensor {} shape: '.format(name), _tensor.shape)
      setattr(self, name, _tensor.to(device))

  @staticmethod
  def rodrigues(r):
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
    m = torch.stack(
      (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
       -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

  @staticmethod
  def with_zeros(x):
    ones = torch.tensor(
      [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret

  @staticmethod
  def pack(x):
    zeros43 = torch.zeros(
      (x.shape[0], x.shape[1], 4, 3), dtype=torch.float64).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret

  def write_obj(self, verts, file_name):
    with open(file_name, 'w') as fp:
      for v in verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

  def forward(self, betas, pose, trans=None, simplify=False):
    batch_num = betas.shape[0]
    id_to_col = {self.kintree_table[1, i]: i
                 for i in range(self.kintree_table.shape[1])}
    parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }
    self.body_shape_blend_shapes =  torch.tensordot(betas, self.shapedirs, dims=([1], [2]))
    v_shaped = self.body_shape_blend_shapes + self.v_template
    J = torch.matmul(self.J_regressor, v_shaped)
    R_cube_big = self.rodrigues(pose.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

    if simplify:
      v_posed = v_shaped
    else:
      R_cube = R_cube_big[:, 1:, :, :]
      I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
                torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=torch.float64)).to(self.device)
      lrotmin = (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
      self.body_pose_blend_shapes = torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))
      v_posed = v_shaped + self.body_pose_blend_shapes

    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )
    for i in range(1, self.kintree_table.shape[1]):
      results.append(
        torch.matmul(
          results[parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
              dim=2
            )
          )
        )
      )

    stacked = torch.stack(results, dim=1)
    results = stacked - \
              self.pack(
                torch.matmul(
                  stacked,
                  torch.reshape(
                    torch.cat((J, torch.zeros((batch_num, 24, 1), dtype=torch.float64).to(self.device)), dim=2),
                    (batch_num, 24, 4, 1)
                  )
                )
              )
    # Restart from here
    self.global_joint_transforms = results
    T = torch.tensordot(results, self.weights, dims=([1], [1]))
    T = T.permute(0, 3, 1, 2)
    rest_shape_h = torch.cat(
      (v_posed, torch.ones((batch_num, v_posed.shape[1], 1), dtype=torch.float64).to(self.device)), dim=2
    )
    v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
    v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
    result = (v + torch.reshape(trans, (batch_num, 1, 3))) if trans is not None else v
    # estimate 3D joint locations
    # print(result.shape)
    # print(self.joint_regressor.shape)
    joints = torch.tensordot(result, self.joint_regressor.T, dims=([1], [0])).transpose(1, 2)
    return result, joints

