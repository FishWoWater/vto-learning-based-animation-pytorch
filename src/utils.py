import os, time

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from scipy import sparse

def load_motion(path, separate_arms=True):
    motion_dict = dict(np.load(path))

    # The recurrent regressor is trained with 30fps sequences
    target_fps = 30
    drop_factor = int(motion_dict["mocap_framerate"] // target_fps)

    trans = motion_dict["trans"][::drop_factor]
    poses = motion_dict["poses"][::drop_factor, :72]
    shape = motion_dict["betas"][:10]

    # Separate arms
    if separate_arms:
        angle = 15
        left_arm = 17
        right_arm = 16

        poses = poses.reshape((-1, poses.shape[-1] // 3, 3))
        rot = R.from_euler('z', -angle, degrees=True)
        poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
        rot = R.from_euler('z', angle, degrees=True)
        poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

        poses = poses.reshape((poses.shape[0], -1))

    # Swap axes
    rotation = R.from_euler("zx", [-90, 270], degrees=True)
    root_rotation = R.from_rotvec(poses[:, :3])
    poses[:, :3] = (rotation * root_rotation).as_rotvec()
    trans = rotation.apply(trans)

    # Remove hand rotation
    poses[:, 66:] = 0

    # Center model in first frame
    trans = trans - trans[0]

    return {
        "pose": poses.astype(np.float32),
        "shape": shape.astype(np.float32),
        "translation": trans.astype(np.float32),
    }


def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()

            if not line_split:
                continue

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    return vertices, faces


def save_obj(filename, vertices, faces):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print("Saved:", filename)


def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2*np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances


def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)


def compute_vertex_normals(vertices, faces):
    # Vertex normals weighted by triangle areas:
    # http://www.iquilezles.org/www/articles/normals/normals.htm

    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    triangles = vertices[faces]

    e1 = triangles[::, 0] - triangles[::, 1]
    e2 = triangles[::, 2] - triangles[::, 1]
    n = np.cross(e2, e1)

    np.add.at(normals, faces[:,0], n)
    np.add.at(normals, faces[:,1], n)
    np.add.at(normals, faces[:,2], n)

    return normalize(normals)


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def faces2edges(F):
    E = set()
    for f in F:
        N = len(f)
        for i in range(N):
            j = (i + 1) % N
            E.add(tuple(sorted([f[i], f[j]])))
    return np.array(list(E), np.int32)


def edges2graph(E):
    G = {}
    for e in E:
        if not e[0] in G: G[e[0]] = {}
        if not e[1] in G: G[e[1]] = {}
        G[e[0]][e[1]] = 1
        G[e[1]][e[0]] = 1
    return G


def laplacianMatrix(F):
    E = faces2edges(F)
    G = edges2graph(E)
    row, col, data = [], [], []
    for v in G:
        n = len(G[v])
        row += [v] * n
        col += [u for u in G[v]]
        data += [1.0 / n] * n
    return sparse.coo_matrix((data, (row, col)), shape=[len(G)] * 2)


def neigh_faces(F, E=None):
    if E is None: E = faces2edges(F)
    G = {tuple(e): [] for e in E}
    for i, f in enumerate(F):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]
    neighF = []
    for key in G:
        if len(G[key]) == 2:
            neighF += [G[key]]
        elif len(G[key]) > 2:
            print("Neigh F unexpected behaviour")
            continue
    return np.array(neighF, np.int32)

def query_closest_vertices(k, q):
    start = time.time()
    tree = cKDTree(k)
    idx = tree.query(q)[1]
    end = time.time()
    print('query finished in {}ms'.format((end - start) * 1000.))
    return idx
