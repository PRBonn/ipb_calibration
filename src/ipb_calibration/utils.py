import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

def np2o3d(points, colors=None):
    scan = o3d.geometry.PointCloud()
    scan.points = o3d.utility.Vector3dVector(points[:, :3])
    if colors is not None:
        scan.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return scan


def get_frame(T, scale=1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1*scale)
    frame.transform(T)
    return frame


def homogenous(x):
    return np.concatenate(
        [x,
         np.ones_like(x[..., :1])],
        axis=-1)


def update_pose(T, dt, dr):
    T_new = np.eye(4)
    DR = Rotation.from_rotvec(dr).as_matrix()
    T_new[:3, :3] = T[:3, :3] @ DR
    T_new[:3, -1] = T[:3, -1] + dt
    return T_new


def skew(v):
    """vector(s) to skew symmetric matrix

    Args:
        v (np.array): [3] or [N,3]

    Returns:
        S (np.array): [3,3] or [N,3,3]
    """
    if len(v.shape) == 1:
        S = np.zeros([3, 3])
        S[0, 1] = -v[2]
        S[0, 2] = v[1]
        S[1, 2] = -v[0]
        return S - S.T
    elif len(v.shape) == 2:
        n = len(v)
        S = np.zeros([n, 3, 3])
        S[..., 0, 1] = -v[..., 2]
        S[..., 0, 2] = v[..., 1]
        S[..., 1, 2] = -v[..., 0]
        return S - S.transpose([0, 2, 1])