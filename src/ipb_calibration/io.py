import yaml
from ipb_calibration import camera, lidar


def sensors_from_dict(cfg):
    """
        Create calibration object from a calibration configuration.
        
        Returns:
            a tuple (cam, lidars) of lists of cameras, and lidars

    """
    cameras = []
    lidars = []

    for topic in cfg:
        if ("image" in topic) or ("cam" in topic):
            cameras.append(camera.Camera.fromdict(cfg[topic]))
        if ("points" in topic) or ("laser" in topic) or ("lidar" in topic):
            lidars.append(lidar.Lidar.fromdict(cfg[topic]))
    return cameras, lidars


def sensors_from_dict_as_dict(cfg):
    """
        Create calibration object from a calibration configuration.
        Returns:
            a tuple (cameras, lidars) of dicts with the names (e.g.topics names) as keys
    """
    cameras = dict()
    lidars = dict()

    for topic in cfg:
        if ("image" in topic) or ("cam" in topic):
            cameras[topic] = camera.Camera.fromdict(cfg[topic])
        if ("points" in topic) or ("laser" in topic) or ("lidar" in topic):
            lidars[topic] = lidar.Lidar.fromdict(cfg[topic])
    return cameras, lidars


def read_calibration_yaml_file(filename:str):
    """
    Read a calibration written as yaml file and initialize corresponding
    laser and camera objects
    
    Args:
        filename: 
            The (complete) filename of the yaml file
    Returns:
        A tuple (cams, lasers, cam_names, laser_names) with
            cams: a dict of camera objects
            lasers: a dict of laser objects
    """
    with open(filename) as f:
        calib = yaml.load(f, Loader=yaml.SafeLoader)
      
    cams, lasers = sensors_from_dict_as_dict(calib)
  
    return (cams, lasers)

