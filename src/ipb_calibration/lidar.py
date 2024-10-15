import numpy as np
from numpy.linalg import norm, inv
from ipb_calibration.utils import update_pose


def transform(points, T_cam_map=np.eye(4), T_os_cam=np.eye(4)):
    points = np.concatenate(
        [points, np.ones_like(points[..., :1])], axis=-1)  # [b,n,4]

    points_t = (T_cam_map @ T_os_cam @ points.T).T
    return points_t


class LinearLidarIntrinsics:
    def __init__(self, bias=0.0, scale=1.0, estimate_bias=True, estimate_scale=True,) -> None:
        self.bias = bias
        self.scale = scale
        self.estimate = np.array([estimate_bias, estimate_scale])
        self.num_params = np.sum(self.estimate)

    @classmethod
    def fromlist(cls, obj):
        return cls(*obj)

    def __repr__(self) -> str:
        return f"LinearLidarIntrinsics(bias: {self.bias},scale: {self.scale})"

    @property
    def params(self):
        return np.concatenate([np.atleast_1d(self.bias), np.atleast_1d(self.scale)])

    def apply(self, scan):
        out = scan * \
            (self.scale + self.bias / (norm(scan, axis=-1, keepdims=True)+1e-8))

        return out
      
    def apply_inverse(self, scan):
        out = scan * \
            ( (1/self.scale) - (self.bias/self.scale) / (norm(scan, axis=-1, keepdims=True)+1e-8))

        return out
      
    def jacobian(self, scan: np.array):
        # first bias, then scale
        J = np.stack(
            [scan / (norm(scan, axis=-1, keepdims=True)+1e-8), scan], axis=-1)  # N,3,2
        return J[:, :, self.estimate]

    def update_params(self, dparams):
        db, ds = np.split(dparams.flatten(), [self.estimate[0]])
        if len(db) == 1:
            self.bias += db
        if len(ds) == 1:
            self.scale += ds


class Lidar:
    def __init__(self, T_lidar_ref=np.eye(4), intrinsics=LinearLidarIntrinsics()):
        self.T_lidar_ref = T_lidar_ref
        self.intrinsics = intrinsics

    @classmethod
    def fromdict(cls, datadict: dict):
        """Initializes a Lidar from a dictionary. Dictionary should have the keys 'extrinsics': [4,4] Transformation matrix, and 'intrinsics' [2,] list with offset and scale. 

        Args:
            datadict (dict): dictionary with keys 'extrinsics' and 'intrinsics'

        Returns:
            Lidar: Instance
        """
        return cls(T_lidar_ref=np.array(datadict["extrinsics"]), intrinsics=LinearLidarIntrinsics.fromlist(datadict['intrinsics']))

    def apply_intrinsics(self, points: np.ndarray):
        """Applys the intrinsics of the lidar to a local point cloud

        Args:
            points (np.ndarray): [n,3] point cloud in lidar frame

        Returns:
            points_out: [n,3] points after applying the intrinsics (e.g. range scale and offset)
        """
        return self.intrinsics.apply(points)
      
    def apply_intrinsics_inverse(self, points: np.ndarray):
        """Applys the inverse intrinsics of the lidar to a local point cloud.
           Thus given a perfect scan with no wrong scale and offset, compute
           a distorted scan.

        Args:
            points (np.ndarray): [n,3] point cloud in lidar frame

        Returns:
            points_out: [n,3] points after inverse applying the intrinsics (e.g. 1/range scale and -offset/scale)
        """
        return self.intrinsics.apply_inverse(points)
      

    def scan2world(self, points: np.ndarray, T_ref_world=np.eye(4)):
        """Transforms a point cloud from the sensor frame over the reference frame into the world frame with applying the intrinsics

        Args:
            points (np.ndarray): [n,3] Point cloud in sensor frame
            T_ref_world (_type_, optional): Transformation from reference frame into the desired frame. Defaults to np.eye(4).

        Returns:
            points_world: [n,3] Point cloud in world frame
        """
        return transform(self.apply_intrinsics(points), T_ref_world, self.T_lidar_ref)[..., :3]

    def __repr__(self) -> str:
        return f"Lidar(T: {self.T_lidar_ref}, intrinsics: {self.intrinsics})"

    @property
    def num_params(self):
        return 6 + self.intrinsics.num_params

    def get_param_dict(self, **kwargs):
        out = {"extrinsics": self.T_lidar_ref,
               "intrinsics": self.intrinsics.params}
        for key, value in kwargs.items():
            out[key] = value
        return out

    def jacobians(self, T_world, scan, normals_corr):
        # Jacobians of Calibration params of lidar
        points_scaled = self.apply_intrinsics(scan)

        de_dt_l = normals_corr@T_world[:3, :3]

        RwT_n = self.T_lidar_ref[:3, :3].T @ T_world[:3,
                                                     :3].T @ normals_corr.T
        de_dr_l = np.cross(points_scaled,
                           RwT_n.T, axis=-1)

        # Jacobians of world pose
        de_dt_w = normals_corr

        de_dr_w = np.cross(transform(points_scaled, self.T_lidar_ref)[..., :3],
                           normals_corr @ T_world[:3, :3], axis=-1)

        # Jacobians of Intrinsic Lidar
        de_dli = self.intrinsics.jacobian(scan)
        de_di = (
            (normals_corr @ T_world[:3, :3] @ self.T_lidar_ref[:3, :3])[:, None, :]@de_dli)[:, 0]

        # Concat Jacobians and add to Normal equations
        # Param Order: t_w, r_w, t_c, r_c, t_l, r_l, p
        J_lidar = np.concatenate(
            [de_dt_l, de_dr_l, de_di],
            axis=-1)
        J_pose = np.concatenate(
            [de_dt_w, de_dr_w], axis=-1)
        return J_pose, J_lidar

    def update_params(self, dparams):
        dt, dr, di = np.split(dparams.flatten(), [3, 6])
        self.T_lidar_ref = update_pose(self.T_lidar_ref, dt, dr)
        self.intrinsics.update_params(di)

        dr, dt = np.linalg.norm(dr), np.linalg.norm(dt)
        return dr, dt
