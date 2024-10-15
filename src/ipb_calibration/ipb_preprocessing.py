from numpy.linalg import norm, inv
import copy
import ipdb
import cv2
from scipy.spatial.transform import Rotation
import open3d as o3d
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from ipb_calibration.apriltag import Apriltags
from numpy.linalg import inv

from diskcache import FanoutCache
from ipb_calibration.utils import homogenous
cache = FanoutCache("cache/", size_limit=3e11)


###################################################################################################
# Open3d lidar based initial guess
###################################################################################################


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(target, sources, voxel_size):
    sources_down, sources_fpfh = [], []
    for source in sources:
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        sources_down.append(source_down)
        sources_fpfh.append(source_fpfh)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return sources, target, sources_down, target_down, sources_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.9999))
    return result


def estimate_initial_guess_lidar(map, scans, voxel_size=0.1, visualize=False, fine_tune=False, max_dist=0.5):
    print("Estimate initial Guess")
    sources, target, sources_down, target_down, sources_fpfh, target_fpfh = prepare_dataset(
        target=map, sources=scans, voxel_size=voxel_size)
    poses = []

    for source, source_down, source_fpfh in zip(tqdm.tqdm(sources), sources_down, sources_fpfh):
        result = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size=voxel_size)
        if fine_tune:
            loss = o3d.pipelines.registration.GMLoss(k=0.1)
            method = o3d.pipelines.registration.TransformationEstimationPointToPlane(
                loss)
            result = o3d.pipelines.registration.registration_icp(
                source, target, max_dist, result.transformation,
                method)
        if visualize:
            draw_registration_result(
                source=source, target=target, transformation=result.transformation)
        poses.append(result.transformation)
    return np.stack(poses, axis=0)


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


@cache.memoize(typed=True)
def estimate_init_lidar_poses(path_data, faro_file, voxel_size=0.1):
    pcd_map = o3d.io.read_point_cloud(faro_file)
    lidar_points, _ = parse_lidar_data(path_data)
    T_lidar_map = estimate_initial_guess_lidar(
        pcd_map, lidar_points[0], voxel_size=voxel_size)
    return T_lidar_map


def load_scan(file):
    points = np.load(file)
    if len(points.dtype) >= 1:
        xyz = np.stack([points["x"], points["y"], points["z"]], axis=-1)
    else:
        xyz = points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(xyz.reshape([-1, 3]), dtype=float))
    return cloud


def parse_lidar_data(path_data, max_scanpoints:int=2500):
    """
      Args:
         max_scanpoints: 
             The scans are randomly downsampled. This argument is the maximal 
             number of points after downsampling. If -1, no downsampling
         
    """
    T_os_cam = []
    lidar_points = []
    cfg = yaml.safe_load(open(join(path_data, "calibration.yaml")))

    for pcd_topic in cfg['point_cloud_topics']:
        topic = pcd_topic.replace('/', '')
        folder = join(path_data, topic)
        scans = sorted(glob.glob(
            f'{folder}/*.npy'))
        print('nr scans:', len(scans))

        init_p = cfg['init_lidari_to_cam0'][topic]
        init_c = np.eye(4)
        init_c[:3, :3] = Rotation.from_euler(
            "xyz", np.array(init_p[:3]), degrees=True).as_matrix()
        init_c[:3, -1] += init_p[3:]
        T_os_cam.append(init_c)

        pcds = []
        for i, scanf in enumerate(tqdm.tqdm(scans)):
            scan = load_scan(scanf)
            if max_scanpoints>0:
                ratio = max_scanpoints/len(scan.points)
                if ratio>1:
                    ratio=1
                scan = scan.random_down_sample(ratio)
            # o3d.visualization.draw_geometries([scan])
            pcds.append(scan)
        lidar_points.append(pcds)
    return lidar_points, T_os_cam, cfg


def xyz_rxryrz2pose(xyz, rxryrz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(
        "xyz", angles=rxryrz, degrees=True).as_matrix()
    T[:3, -1] = xyz
    return T
###################################################################################################
# Init guess using Cameras
###################################################################################################


def parse_intrinsics(cam_intrinsics_xml, num_cams=4):
    path_intrinsics = Path(cam_intrinsics_xml).parent
    tree = ET.parse(cam_intrinsics_xml)
    intrinsics = tree.findall("PinholeCamera")
    calib = []
    for i in range(num_cams):
        K = np.eye(3)
        K[[0, 1], [0, 1]] = float(
            intrinsics[i].find("cameraConstant").text)
        K[:2, -1] = np.array([float(xh) for xh in (
            intrinsics[i].find("principalPoint").text.strip().split(" "))])
        ilut = intrinsics[i].find(
            "filenameLutDist2Undist").text.strip()
        lut = intrinsics[i].find(
            "filenameLutUndist2Dist").text.strip()
        calib.append({"K": K, "ilut": join(path_intrinsics, ilut),
                     "lut": join(path_intrinsics, lut)})
    return calib



def est_pose_from_obs(observations, cam_is_pinhole, K=np.eye(3), dist_coeff=np.zeros(5)):
    points_3d = np.concatenate([det["coords"]
                                for det in observations])
    points_2d = np.concatenate([det["corners"]
                           for det in observations])
    if cam_is_pinhole:
        rays = homogenous(points_2d) @ inv(K).T
        rays /= rays[:, -1:]
    else:
        points_2d_undist = cv2.fisheye.undistortPoints(points_2d[:, np.newaxis, :], K, np.zeros((1, 4)))
        points_2d_undist = points_2d_undist.squeeze(1)
        rays = homogenous(points_2d_undist)
        rays /= rays[:, -1:]
        
    valid, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, rays[:, :2], np.eye(3), dist_coeff)
    assert valid
    T_map2cam = np.eye(4)
    T_map2cam[:3, -1:] = tvec
    T_map2cam[:3, :3] = Rotation.from_rotvec(rvec.squeeze()).as_matrix()
    T_cam2map = inv(T_map2cam)
    return T_cam2map, valid, len(inliers)


def estimate_initial_guess(observations, cam_is_pinhole, init_K=None, dist_coeff=None):
    num_cameras = len(observations)
    num_poses = len(observations[0])
    # count for each camera and each timestamp how many observations
    num_obs = np.zeros([num_cameras, num_poses])  # [num_cams, num_poses]
    for c, c_obs in enumerate(observations):
        for t, t_obs in enumerate(c_obs):
            num_obs[c, t] = len(t_obs)

    # Estimate extrinsics between cameras from the timestamp with the most tags in both cameras
    best_pose_t = np.zeros(num_cameras-1, dtype=np.int64)
    for i in range(1, len(observations)):
        best_pose_t[i-1] = np.argmax(np.min(num_obs[[0, i]], axis=0))

    T_cami_cam0 = np.zeros([num_cameras, 4, 4])
    T_cami_cam0[0] = np.eye(4)
    for c_, t in enumerate(best_pose_t):
        c = c_+1
        T_cam0_map, valid_, num_inlier = est_pose_from_obs(
            observations[0][t], cam_is_pinhole[0], init_K[0], dist_coeff[0])
        T_cami_map, valid_, num_inlier = est_pose_from_obs(
            observations[c][t], cam_is_pinhole[c], init_K[c], dist_coeff[c])
        T_cami_cam0[c] = inv(T_cam0_map) @ T_cami_map

    # Estimate pose from the image with most tags
    best_cam_t = np.argmax(num_obs, axis=0)

    T_cami_map_t = np.zeros([num_poses, 4, 4])
    for t, c in enumerate(best_cam_t):
        T_cam2map, valid_, num_inlier = est_pose_from_obs(
            observations[c][t], cam_is_pinhole[c], init_K[c], dist_coeff[c])
        assert valid_, f"NOT VALID POSE ESTIMATION, num inlier: {num_inlier}"
        T_cami_map_t[t] = T_cam2map
    T_cam0_map_t = T_cami_map_t @ inv(T_cami_cam0[best_cam_t])

    return T_cam0_map_t, T_cami_cam0


def parse_camera_data(apriltag_file, path_data):
    april = Apriltags(apriltag_file)
    cfg = yaml.safe_load(open(join(path_data, "calibration.yaml")))

    num_cams = len(cfg["image_topics"])
    observations = []
    T_cami_cam0 = []

    image_sizes = np.zeros([num_cams, 2], dtype=np.int64)
    seen_tags = {}
    for c, image_topic in enumerate(tqdm.tqdm(cfg['image_topics'])):
        observations.append([])
        topic = image_topic.replace('/', '')
        print(10*"*", topic, 10*"*")

        folder = join(path_data, topic)

        images = sorted(glob.glob(
            f'{folder}/*.png'))

        for t, image in enumerate(tqdm.tqdm(images)):
            img = plt.imread(image)
            image_sizes[c] = img.shape[1], img.shape[0]
            gray = (np.mean(img, axis=-1)*255).astype("uint8")
            observations[-1].append(april.detect(gray))
            for det in observations[-1][-1]:
                seen_tags[det["tag_id"]] = det["coords"]
        
        if ('init_cami_to_cam0' in cfg.keys()):
            init_p = cfg['init_cami_to_cam0'][topic]
            init_c = np.eye(4)
            init_c[:3, :3] = Rotation.from_euler(
                "xyz", np.array(init_p[:3]), degrees=True).as_matrix()
            init_c[:3, -1] += init_p[3:]
            T_cami_cam0.append(init_c)

    tag, coords = np.fromiter(
        seen_tags.keys(), dtype=np.int64), np.stack([*seen_tags.values()])
    coords = coords.reshape([-1, 3])
    tag2indx = np.full([tag.max()+1, 4], -1)
    tag2indx[tag] = np.arange(len(tag)*4).reshape([-1, 4])

    indices = []
    imgpixel = []
    for c, c_obs in enumerate(observations):
        c_ind = []
        c_img = []
        for t, t_obs in enumerate(c_obs):
            if len(t_obs) > 0:
                img_pixel = np.concatenate(
                    [d["corners"] for d in t_obs], axis=0)
                idx = tag2indx[[d["tag_id"] for d in t_obs]].reshape(-1)
            else:
                img_pixel = np.zeros([0, 2])
                idx = np.zeros([0])
            c_ind.append(idx)
            c_img.append(img_pixel)

        imgpixel.append(c_img)
        indices.append(c_ind)
    
    # Take a look if user has given camera model in yaml file
    if 'camera_models' in cfg.keys():
        cam_is_pinhole = [m == 'pinhole' for m in cfg['camera_models']]
        # Check
        cam_is_fisheye = [m == 'fisheye' for m in cfg['camera_models']]
        if np.sum(cam_is_pinhole)+np.sum(cam_is_fisheye) != len(T_cami_cam0):
            print('\n**************************************************************************************')
            print('WARNING: The sum of all pinhole and all fisheye cameras is not the number of all cameras !!')
            print('         Did you use the wrong keyword for camera models in calibration.yaml ?')
            print('**************************************************************************************',
                  flush=True)
    else:
        print('\n\nNo camera models given in yaml file, assuming pinhole for all cameras.\n', flush=True)
        cam_is_pinhole = len(T_cami_cam0) * [True]
        
    return imgpixel, indices, coords, observations, T_cami_cam0, image_sizes, cam_is_pinhole

###################################################################################################
# Init guess Camera intrinsics
###################################################################################################


def find_plane(pts3d, pts2d, threshold=0.1):
    num = []
    points3d = []
    points2d = []
    for (p1, p2, p3, p4) in pts3d:
        e1 = (p2-p1)/norm(p2-p1)
        e2 = (p3-p1)/norm(p3-p1)
        e3 = np.cross(e1, e2)
        R = np.stack([e1, e2, e3], axis=-1)
        pts_loc = (pts3d-p1) @ R
        valids = np.all(np.abs(pts_loc[:, :, -1]) < threshold, axis=-1)
        valid_pts = pts_loc[valids]
        valid_pts = valid_pts.reshape(-1, 3)
        valid_pts[:, -1] = 0

        points3d.append(valid_pts)
        points2d.append(pts2d[valids].reshape(-1, 2))
        num.append(len(valid_pts))

    best = np.argmax(num)
    return points3d[best], points2d[best]


def estimate_initial_k(observations, cam_is_pinhole, max_z_dist=0.1, image_size=(2064, 1024), max_num_images=5):
    num_cameras = len(observations)
    num_poses = len(observations[0])

    max_num_images = min(
        max_num_images, num_poses) if max_num_images > 0 else num_poses
    # count for each camera and each timestamp how many observations
    num_obs = np.zeros([num_cameras, num_poses])  # [num_cams, num_poses]
    init_values = []
    init_coeff = []
    for c, c_obs in enumerate(observations):

        image_points = [np.zeros([0, 2]) for _ in range(max_num_images)]
        object_points = [np.zeros([0, 3]) for _ in range(max_num_images)]
        image_size_c = image_size if isinstance(
            image_size, tuple) else tuple(image_size[c])
        for t, t_obs in enumerate(c_obs):
            if len(t_obs) == 0:
                continue
            pts3d = np.stack([det["coords"]
                              for det in t_obs])
            pts2d = np.stack([det["corners"]
                              for det in t_obs])
            points_3d, points_2d = find_plane(
                pts3d, pts2d, threshold=max_z_dist)
            num_obs[c, t] = len(points_3d)

            num_pts = [len(i) for i in object_points]
            min_idx = np.argmin(num_pts)

            if len(points_3d) >= num_pts[min_idx]:
                object_points[min_idx] = points_3d.astype(np.float32)
                image_points[min_idx] = points_2d.astype(np.float32)

        print(f"cam{c}, num pts", num_obs[c, num_obs[c] >= 12])
        if cam_is_pinhole[c]:
            retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, image_size_c, None, None,
            )
        else:
            calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC 

            # fisheye calibration is more stable when skew and distortion parameters are kept at zero
            calibration_flags += cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_FIX_K1 \
                                + cv2.fisheye.CALIB_FIX_K2 + cv2.fisheye.CALIB_FIX_K3 + cv2.fisheye.CALIB_FIX_K4

            retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                        [obj_points[:, np.newaxis, :] for obj_points in object_points], \
                            [img_points[:, np.newaxis, :] for img_points in image_points], \
                                image_size_c, None, None, flags=calibration_flags)
    
            distCoeffs = np.zeros((1, 5)) # set distCoeffs to zero as we use other distortion model as openCV 
        print("Intrinsics reprojection error: ", retval)
        init_k = cameraMatrix[[0, 1, 0, 1], [2, 2, 0, 1]]
        init_values.append(init_k)
        init_coeff.append(distCoeffs)
    return np.stack(init_values, axis=0), np.concatenate(init_coeff, axis=0),

###################################################################################################
