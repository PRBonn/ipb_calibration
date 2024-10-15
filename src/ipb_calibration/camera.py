import os
from numpy.linalg import norm
import ipdb
from numpy.linalg import inv
import numpy as np
from ipb_calibration.utils import homogenous, skew, update_pose
import cv2
from scipy import interpolate


def inner(a, b):
    return np.einsum("...d,...d->...", a, b)


def outer(a, b):
    return np.einsum("...b,...d->...bd", a, b)


def diag(a):
    return np.einsum("...d,db->...db", a, np.eye(a.shape[-1]))


def batch_eye(n, d):
    return np.zeros([n, d, d]) + np.eye(d)


class CVDistortionModel:
    def __init__(self, degree=3, division_model=True, cv2_coeff=None) -> None:
        radial_degree = degree if not division_model else 0
        division_degree = degree if division_model else 0

        self.k = np.zeros(radial_degree)
        self.h = np.zeros(division_degree)
        self.p = np.zeros(2)

        if cv2_coeff is not None:
            assert len(cv2_coeff) >= 5
            idx = np.arange(min(len(cv2_coeff), 8))
            idx[:5] = [0, 1, 4, 2, 3]
            params = np.array(cv2_coeff)[idx]
            k, p, h = np.split(params, [3, 5])
            if len(cv2_coeff) < 8:
                h = -k

            max_idx = min(3, len(self.k))
            self.k[:max_idx] = k[:max_idx]

            max_idx = min(3, len(self.h))
            self.h[:max_idx] = h[:max_idx]

            self.p = p

    def __str__(self) -> str:
        return f"CVDistortion (k: {self.k}, h: {self.h}, p: {self.p})"

    @classmethod
    def fromcvlist(cls, cv2_coeff):
        params = np.array(cv2_coeff).squeeze()[:8]
        assert len(params) >= 5
        params[:5] = params[[0, 1, 4, 2, 3]]
        k, p, h = np.split(params, [3, 5])
        out = cls()
        # up the one from which on all are zero
        out.k = k[(1 - np.cumprod(k[::-1] == 0)[::-1]).astype('bool')]
        out.h = h[(1 - np.cumprod(h[::-1] == 0)[::-1]).astype('bool')]
        out.p = p
        return out

    @property
    def radial_degree(self):
        return len(self.k)

    @property
    def division_degree(self):
        return len(self.h)

    @property
    def params(self):
        return np.concatenate([self.k, self.h, self.p])

    def distort(self, x):
        k = np.concatenate([[1], self.k])  # [degree+1]
        h = np.concatenate([[1], self.h])  # [degree+1]
        p = self.p

        d = inner(x, x)  # d= r^2 [n]

        d_pol = d[:, None]**np.arange(len(k))  # [n,degree+1]
        d_pol2 = d[:, None]**np.arange(len(h))  # [n,degree+1]
        S = np.eye(2)[[1, 0]]

        Sp = (p @ S)[None, :]

        dx_tangential = 2 * p * \
            np.prod(x, axis=-1, keepdims=True) + \
            Sp * d[:, None] + \
            2 * Sp * x**2
        x_dist = x * inner(k, d_pol)[:, None] / \
            inner(h, d_pol2)[:, None] + dx_tangential
        return x_dist

    def jacobian(self, x: np.array):
        p = self.p
        k = np.concatenate([[1], self.k])  # [degree+1]
        h = np.concatenate([[1], self.h])  # [degree+1]
        S = np.eye(2)[[1, 0]]

        d = inner(x, x)  # d= r^2 [n]
        d_pol = d[:, None]**np.arange(len(k))  # [n,degree+1]
        d_pol2 = d[:, None]**np.arange(len(h))  # [n,degree+1]

        # ipdb.set_trace()
        dpol_dx = np.sum(np.arange(1, len(k)) *
                         k[1:] * d_pol[:, :-1], axis=-1)[:, None, None]  # [n,1,1]
        dpol2_dx = np.sum(np.arange(1, len(h)) *
                          h[1:] * d_pol2[:, :-1], axis=-1)[:, None, None]  # [n,1,1]

        sum_kd = inner(k, d_pol)[:, None, None]
        sum_hd = inner(h, d_pol2)[:, None, None]
        # derivative of point
        dxdist_dx = np.eye(2)[None, :] * sum_kd / sum_hd
        dxdist_dx += 2 * outer(x, x) * dpol_dx / sum_hd  # radial
        dxdist_dx -= 2 * outer(x, x) * dpol2_dx / sum_hd**2  # division
        dxdist_dx += 2 * outer(p, x @ S)
        dxdist_dx += 2 * outer(S @ p, x)
        dxdist_dx += 4 * S @ diag(p*x)

        # derivative of p
        dxdist_dp = 2*np.eye(2)[None, :] * np.prod(x, axis=-1)[:, None, None]
        dxdist_dp += S[None, :] * d[:, None, None]
        dxdist_dp += 2 * S[None, :] * x[:, :, None]**2

        # derivative of k
        dxdist_dk = x[:, :, None] * d_pol[:, None, 1:] / sum_hd
        # derivative of h
        dxdist_dh = -x[:, :, None] * sum_kd * d_pol2[:, None, 1:]/(sum_hd**2)

        dxdist_dparams = np.concatenate(
            [dxdist_dk, dxdist_dh, dxdist_dp], axis=-1)
        return dxdist_dx, dxdist_dparams

    def undistort(self, p_dist, num_iter=20):
        p = np.copy(p_dist)
        for i in range(num_iter):
            error = self.distort(p) - p_dist  # [n,2]
            error = error.reshape([len(p), 2, 1])

            J, _ = self.jacobian(p)  # [n,2,2]
            JT = J.transpose([0, 2, 1])
            N = (JT @ J)
            g = -JT @ error

            dp = inv(N) @ g
            p += dp[:, :, 0]
        return p

    def update_params(self, dparams):
        dks, dhs, dps = np.split(dparams.flatten(), np.cumsum(
            [self.radial_degree, self.division_degree]))
        self.k += dks
        self.h += dhs
        self.p += dps

    def to_cv2(self):
        params = np.zeros(8)
        deg = min(2, self.radial_degree)
        params[:deg] = self.k[:deg]
        params[2:4] = self.p
        if self.radial_degree >= 3:
            params[4] = self.k[2]
        deg = min(3, self.division_degree)
        params[5:5+deg] = self.h[:deg]
        return params


class Camera:
    def __init__(self, K=np.eye(3), T_cam_ref=np.eye(4), distortion=CVDistortionModel(), is_pinhole=True):
        self.T_cam_ref = T_cam_ref
        self.distortion = distortion
        self.K = K
        self.is_pinhole = is_pinhole
        self.projection = PinholeProjection if is_pinhole else FisheyeProjection

    @classmethod
    def fromdict(cls, datadict: dict):
        """Initializes a camera from a dictionary. Dictionary should have the keys 'extrinsics': [4,4] Transformation matrix, 'distortion_coeff' [8,] list in cv2 format, and 'K' [3,3] camera matrix 

        Args:
            datadict (dict): dictionary with keys 'extrinsics' and 'distortion_coeff' and 'K' 

        Returns:
            Camera: Instance
        """
        return cls(K=np.array(datadict["K"]),
                   T_cam_ref=np.array(datadict["extrinsics"]),
                   distortion=CVDistortionModel.fromcvlist(datadict["distortion_coeff"]),
                   is_pinhole = datadict["is_pinhole"])

    def __str__(self) -> str:
        return f"Camera: (Projection: {'Pinhole' if self.is_pinhole else 'Fisheye'},K: {self.K}, T_cam_ref: {self.T_cam_ref}, distortion: {self.distortion})"

    def __repr__(self) -> str:
        return self.__str__()

    def undistort(self, p_img: np.ndarray, num_iter=10, Kresult=None):
        """Rectifies the coordinates p_img.

        Args:
            p_img (np.ndarray): 
                [n,2] Coordinates in raw image
            num_iter (int, optional): 
                Number iterations to solve the non linear least squares problem of the rectification. Defaults to 20.
            Kresult: 
                the 3x3 calibration matrix which is valid for the final resulting coordinates
                AND the input coordinates. If None,
                the K stored in this class is used (which is usually the estimated K)

        Returns:
            p_rect: [n,2] rectified coordinates
        """
        if Kresult is None:
            Kresult = self.K
          
        x_d = (homogenous(p_img) @ inv(Kresult).T)[:, :-1]  # [n,2]
        x_n = self.distortion.undistort(x_d, num_iter=num_iter)
        r = homogenous(x_n)
        r = r @ Kresult.T
        return r[:, :2]
      
    def distort(self, p_img: np.ndarray, Kresult=None):
        """Calculate distorted image coordinates from distortion free ones.

        Args:
            p_img (np.ndarray): 
                [n,2] Coordinates in distortion free image
            Kresult: 
                the 3x3 calibration matrix which is valid for the final resulting coordinates
                AND the input coordinates. If None,
                the K stored in this class is used (which is usually the estimated K)

        Returns:
            p_rect: [n,2]  coordinates in distorted image
        """
        if Kresult is None:
            Kresult = self.K

        x_d = (homogenous(p_img) @ inv(Kresult).T)[:, :-1]  # [n,2]
        x_n = self.distortion.distort(x_d)
        r = homogenous(x_n)
        r = r @ Kresult.T
        return r[:, :2]

    def projection_valid_mask(self, pts_world: np.ndarray, img_size, T_ref_world=np.eye(4)):
        """ Projects 3d Points from the world system over the reference system into the image frame

        Args:
            pts_world (np.ndarray): [n,3] Points in world frame
            T_ref_world (np.ndarray, optional): Transformation from reference frame to the world frame. Defaults to np.eye(4).
        Returns:
            pts_im: [N,2] Pixel coordinates in raw (distorted) image 
        """
        x_c = ((inv(self.T_cam_ref) @ inv(T_ref_world) @
                homogenous(pts_world).T)[:3]).T  # [3,N]
        in_front = x_c[:, -1] > 0
        x_n = x_c[:, :2]/x_c[:, -1:]

        border_i = np.array([[0.0, 0],
                             [img_size[1], img_size[0]]])
        border_d = (homogenous(border_i) @ inv(self.K).T)[:, :-1]  # [n,2]
        border_n = self.distortion.undistort(border_d)

        inside_view = np.all(x_n > border_n[None, 0], axis=-1) & np.all(x_n <
                                                                        border_n[None, 1], axis=-1)

        x_d = self.distortion.distort(x_n)
        x_i = x_d @ self.K[:2, :2] + self.K[None, :2, -1]
        inside_image = np.all(x_i > 0, axis=-1) & np.all(x_i <
                                                         np.array(img_size[:2])[None, ::-1]-1, axis=-1)
        return in_front & inside_image  & inside_view

    def project(self, pts_world: np.ndarray, T_ref_world=np.eye(4)):
        """ Projects 3d Points from the world system over the reference system into the image frame

        Args:
            pts_world (np.ndarray): [n,3] Points in world frame
            T_ref_world (np.ndarray, optional): Transformation from reference frame to the world frame. Defaults to np.eye(4).
        Returns:
            pts_im: [N,2] Pixel coordinates in raw (distorted) image 
        """
        x_c = ((inv(self.T_cam_ref) @ inv(T_ref_world) @
                homogenous(pts_world).T)[:3]).T  # [3,N]
        x_n = self.projection.project(x_c)
        x_d = self.distortion.distort(x_n)
        x_i = x_d @ self.K[:2, :2] + self.K[None, :2, -1]
        return x_i

    def undistort_image(self, image_distorted: np.ndarray, method="linear", Kresult=None):
        """ Undistort an image
            Args:
                Kresult: 
                    the 3x3 calibration matrix which is valid for the final image. If None,
                    the K stored in this class is used (which is usually the estimated K)
        """
        if Kresult is None:
            Kresult = self.K

        size = image_distorted.shape

        size = (size[1], size[0])
        yy, xx = np.meshgrid(
            np.arange(size[1], dtype=np.float64), np.arange(size[0]), indexing="ij")
        pts = np.stack([xx, yy], axis=-1)
        pts_list = pts.reshape([-1, 2])

        x_c = homogenous(pts_list) @ inv(Kresult).T
        x_n = x_c[:, :2]/x_c[:, -1:]
        x_d = self.distortion.distort(x_n)
        undist_list = x_d @ self.K[:2, :2] + self.K[None, :2, -1]

        coords = undist_list.reshape(pts.shape).clip(
            0, max=[size[0]-1, size[1]-1])
        interp = interpolate.RegularGridInterpolator((
            np.arange(size[1], dtype=np.float64), np.arange(size[0])), image_distorted, bounds_error=False, fill_value=None, method=method)
        image_undistorted = interp(coords[..., [1, 0]])

        return image_undistorted
      
    def calculate_lookup_tables(self, image_size, Kresult=None):
        """
          Calculate lookup tables, like used in calibration software tcc. A calibration
          matrix which is valid for the final lookuptables can be given.
          The calculation/adaption follows the follwing idea:
            The distorted coordinates with the two different Ks should be
            the same (this is for tcc distortion modelling!):
              x_dist_o = K * xn + lut(K*xn)
              x_dist_r = Kr*xn +  lutr(Kr*xn)
            If you set x_dist_o == x_distr:
              lutr(Kr*xn) = -Kr*xn + K*n + lut(K*xn)
              
            The distorion functions here are always absolute, thus:
              lut(x) = dist(x) - x
              
            If you apply this to the above function:
              lutr(Kr*xn) = dist(K*xn) - Kr*xn
              
            If dist(x, Kr) = Kr * d(inv(Kr) * x), then you have to
            apply a correction:
              lutr(Kr*xn) = K*inv(Kr) * dist(Kr*xn, Kr) - Kr*xn
          
          Args:
            self: 
              The camera object 
            image_size: 
              a numpy array or list of length 2 with [no_of_columns, no_of_rows]
              of the image.
            Kresult: 
              the 3x3 calibration matrix which is valid for the final image. If None,
              the K stored in this class is used (which is usually the estimated K)
              
          Returns:
            (lut, ilut):
              Two 2 x columns x rows float arrays with offset values.
              lut[0,:,:] are the offsets in column direction,
              lut[1,:,:] in row direction.
              Correction is is always done with new_coord = old_coord + lut
              x_distorted = x_distortion_free + lut
              x_distortion_free = x_distorted + ilut
        """
        # All coordinates of the image  
        p_img_2D = np.mgrid[range(image_size[0]), range(image_size[1])]
        p_img = np.reshape(p_img_2D.T, (image_size[0]*image_size[1], 2))
        
        # For the lut: Given coordinates in the distortion free image, get offsets for the coordinate in
        # the original = raw = distorted image
        if Kresult is None:
          M = np.eye(3)
        else:
          M =  self.K @ inv(Kresult)

        p_dist = self.distort(p_img, Kresult)
        p_dist = (homogenous(p_dist) @ M.T)[:, :-1]
        lut_1D = p_dist - p_img
        lut = np.reshape(lut_1D, (image_size[1], image_size[0], 2)).T
        
        # For the ilut: Given coordinates in the original = distorted image, get offsets for the coordinates in
        # the distortion free image
        p_corr = (homogenous(p_img) @ inv(M).T)[:, :-1]
        p_free = self.undistort(p_corr, Kresult=Kresult)
        ilut_1D = p_free - p_img
        ilut = np.reshape(ilut_1D, (image_size[1], image_size[0], 2)).T
        
        return (lut, ilut)

    @staticmethod 
    def write_tcc_lut_file(lut_filename: str, lut: np.ndarray):
        """ 
        Write a lut or ilut filename in the format of the calibration software tcc
        
        Args:
          lut_filename: 
            the (complete) filename
          lut: A 2 x columns x rows float array. lut[0,:,:] are the offsets in column direction
               lut[1,:,:] in row direction (correction in tcc is always done with new_coord = old_coord + lut)
        """
        with open(lut_filename,'w') as fid:
        
            # Header
            fid.write('# Filename:%s\n' % os.path.basename(lut_filename))
            fid.write('#\n')
            fid.write('distortiontable\n')
            fid.write('# TCC LUT Table\n')
            fid.write('#\n')
            fid.write('# rectangle for array\n')
            fid.write('# basex basey dimx dimy\n')
            fid.write('   0    0 %d %d\n' % (lut.shape[1], lut.shape[2]))
            fid.write('#\n')
            
            # Values
            for y in range(lut.shape[2]):
                fid.write('##-----------------------------dx dy for row:    %d\n' % y);
                for x in range(lut.shape[1]):
                    fid.write('  %f  %f\n' % (lut[0, x, y], lut[1, x, y]) );

    @property
    def num_params(self):
        return 6 + 4 + len(self.distortion.params)

    def get_param_dict(self, **kwargs):
        out = {"extrinsics": self.T_cam_ref, "K": self.K,
               "distortion_coeff": self.distortion.to_cv2(),
               "is_pinhole": self.is_pinhole}
        for key, value in kwargs.items():
            out[key] = value
        return out

    def pix2ray(self, x_i):
        x_d = (homogenous(x_i) @
               np.linalg.inv(self.K).T)[:, :-1]  # [n,2]
        x_n = self.distortion.undistort(x_d)
        r = self.projection.to_ray(x_n)
        return r

    def update_params(self, dparams):
        dt, dr, dk, ddist = np.split(dparams.flatten(), np.cumsum(
            [3, 3, 4]))
        self.distortion.update_params(ddist)
        self.T_cam_ref = update_pose(self.T_cam_ref, dt, dr)

        self.K[0, 2] += dk[0]
        self.K[1, 2] += dk[1]
        self.K[0, 0] += dk[2]
        self.K[1, 1] += dk[3]

        dr, dt = np.linalg.norm(dr), np.linalg.norm(dt)
        return dr, dt

    def jacobians(self, T_world, coords, ray2apriltag_idx, num_tags):
        # Help matrices
        T_calib = self.T_cam_ref

        num_rays = len(coords)
        x_r = ((inv(T_world) @
                homogenous(coords).T)[:3]).T  # [N,3] Coordinates in reference
        x_c = ((inv(T_calib) @ inv(T_world) @
                homogenous(coords).T)[:3]).T  # [N,3] Coordinates in cami
        x_n = self.projection.project(x_c)
        x_d = self.distortion.distort(x_n)

        # dpi_dpe
        dxi_dxd = self.K[:2, :2]
        dxd_dxn, dxd_ddistparams = self.distortion.jacobian(x_n)
        dxn_dxc = self.projection.jacobian(x_c)
        dxc_dxr = T_calib[:3, :3].T
        dxr_dxw = T_world[:3, :3].T

        dxi_dxc = dxi_dxd @ dxd_dxn @ dxn_dxc
        dxi_dxr = dxi_dxc @ dxc_dxr

        # Jacobians of Extrinsic Calibration params of camera
        de_dt_c = - dxi_dxc @ T_calib[:3, :3].T

        vec = (x_r-T_calib[None, :3, -1]) @ T_calib[:3, :3]
        de_dr_c = dxi_dxc @ skew(vec)

        # Jacobians of world pose
        de_dt_w = -dxi_dxr @  T_world[:3, :3].T

        vec = (coords-T_world[None, :3, -1]) @ T_world[:3, :3]
        de_dr_w = dxi_dxr @ skew(vec)

        # Jacobians of apriltags
        de_dp_w = np.zeros([num_rays, 2, num_tags, 3])
        for ri in range(num_rays):
            de_dp_w[ri, :, ray2apriltag_idx[ri]
                    ] = dxi_dxr[ri]  @ dxr_dxw
        de_dp_w = de_dp_w.reshape([num_rays, 2, num_tags*3])

        # Jacobians of Intrinsic Camera
        dxi_dxk = np.zeros([num_rays, 2, 4])
        dxi_dxk[:, 0, 0] = 1
        dxi_dxk[:, 1, 1] = 1
        dxi_dxk[:, 0, 2] = x_d[:, 0]
        dxi_dxk[:, 1, 3] = x_d[:, 1]

        de_dk = dxi_dxk  # dpi_dk

        # Jacobians of Camera distortion
        de_dcd = dxi_dxd @ dxd_ddistparams

        # Concat Jacobians and add to Normal equations
        J_cam = np.concatenate([de_dt_c,
                                de_dr_c,
                                de_dk,
                                de_dcd], axis=-1).reshape([num_rays*2, -1])
        J_pose = np.concatenate([de_dt_w,
                                 de_dr_w], axis=-1).reshape([num_rays*2, -1])
        J_tags = de_dp_w.reshape([2*num_rays, -1])
        return J_pose, J_cam, J_tags


###########################################
# Projection
###########################################


class PinholeProjection:
    @staticmethod
    def project(x_c: np.ndarray):
        """project points onto

        Args:
            x_c (np.ndarray): [N,3] points (rays) in camera frame

        Returns:
            x_n: [N,2] projected points
        """
        x_n = x_c[:, :2]/x_c[:, -1:]
        return x_n

    @staticmethod
    def to_ray(x_n):
        """points to rays 

        Args:
            x_n (np.ndarray): [N,2] points in camera frame

        Returns:
            x_c: [N,3] rays in camera system
        """
        r = homogenous(x_n)
        r = r / norm(r, axis=-1, keepdims=True)
        return r

    @staticmethod
    def jacobian(x_c: np.ndarray):
        """Jacobian computation

        Args:
            x_c (np.ndarray): [N,3] points in camera frame

        Returns:
            J: [N,2,3] Jacobians of projection
        """
        J = np.concatenate(
            [np.repeat(np.eye(2)[None, :, :], len(x_c), axis=0), -x_c[:, :2, None]/x_c[:, -1, None, None]], axis=-1)/x_c[:, -1, None, None]
        return J


class FisheyeProjection:
    @staticmethod
    def project(x_c: np.ndarray):
        """project points onto

        Args:
            x_c (np.ndarray): [N,3] points (rays) in camera frame

        Returns:
            x_n: [N,2] projected points
        """
        r_xy = np.sqrt(x_c[:, 0]**2 + x_c[:, 1]**2)
        c_ideal = (x_c[:, 0]/r_xy) * np.arctan2(r_xy, x_c[:, 2])
        r_ideal = (x_c[:, 1]/r_xy) * np.arctan2(r_xy, x_c[:, 2])
        x_n=  np.stack([c_ideal, r_ideal], axis=-1)
        return x_n

    @staticmethod
    def to_ray(x_n):
        """points to rays 

        Args:
            x_n (np.ndarray): [N,2] points in camera frame

        Returns:
            x_c: [N,3] rays in camera system
        """

        r_xy = np.sqrt(x_n[:, 0]**2 + x_n[:, 1]**2)
        X = (np.sin(r_xy)/r_xy) * x_n[:, 0]
        Y = (np.sin(r_xy)/r_xy) * x_n[:, 1]
        Z = np.cos(r_xy)
        return np.stack([X, Y, Z], axis=-1)
    @staticmethod
    def jacobian(x_c: np.ndarray):
        """Jacobian computation

        Args:
            x_c (np.ndarray): [N,3] points in camera frame

        Returns:
            J: [N,2,3] Jacobians of projection
        """
        r_xy = np.sqrt(x_c[:, 0]**2 + x_c[:, 1]**2)
        J = np.zeros([len(x_c), 2, 3])
        J[:, 0, 0] = (1/r_xy - x_c[:, 0]**2/r_xy**3) * np.arctan2(r_xy, x_c[:, 2]) + \
            (x_c[:, 0]**2/x_c[:, 2])/(r_xy**2 * (1+r_xy**2/x_c[:, 2]**2))
        J[:, 0, 1] = x_c[:, 0]*x_c[:, 1]*(1/(x_c[:, 2]*r_xy**2*(1 + r_xy **
                                                        2/x_c[:, 2]**2)) - np.arctan2(r_xy, x_c[:, 2])/r_xy**3)
        J[:, 0, 2] = (-x_c[:, 0]/x_c[:, 2]**2)/(1+r_xy**2/x_c[:, 2]**2)

        J[:, 1, 0] = J[:, 0, 1]
        J[:, 1, 1] = (1/r_xy - x_c[:, 1]**2/r_xy**3) * np.arctan2(r_xy, x_c[:, 2]) + \
            (x_c[:, 1]**2/x_c[:, 2])/(r_xy**2 * (1+r_xy**2/x_c[:, 2]**2))
        J[:, 1, 2] = (-x_c[:, 1]/x_c[:, 2]**2)/(1+r_xy**2/x_c[:, 2]**2)
        return J
