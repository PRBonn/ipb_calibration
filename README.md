# Perception System Calibration

This repository contains the full setup for the system calibration (relative
poses between the sensors + intrinsics) for multipe Cameras and/ or LiDAR sensors.

## Prerequisites

The first thing is to obtain the point cloud and Apriltag coordinates from the calibration environment.
Next we have to record with the setup that should be recorded. After recording we can use this repository for calibration.

### TLS point cloud + Apriltag coordinate extraction

We provide exemplary data here:
- Point Cloud and Apriltag Coordinates: [Download here](https://www.ipb.uni-bonn.de/html/projects/ipb_calibration/reference_data.zip)

#### Using own data

If you want to use your own data: Take a TLS and scan the room. Then use the apriltag coordinate extraction software we provide in this repository. This will
extract 3D coordinates of the apriltags. Use these coordinates later as reference data in folder `reference`. A detailed
description for the apriltag extraction is provided [here](README_apriltag_extraction.md).

For the system calibration: To reduce the compute time I suggest using cloud compare to downsample to half a centimeter resolution and precompute the normals. We assume the point cloud to be in PLY file format.

### Data recording

We need for the calibration from each sensor a certain amount of observations (e.g., images and point clouds) in the calibration environment. For this, one should measure in a stop-and-go manner, i.e., (i) move, (ii) stand still, (iii) take a picture or point cloud from each sensor, repeat (i) - (iii) to have around 50 observations from each sensor. In the end it would be nice to be moved around 360 degrees with the setup such that each sensor saw most of the environment. When recording with 2D profile scanner, make sure that they see enough structural elements otherwise the calibration might fail. While taking the pictures/ point cloud please avoid blocking the sensors (try to find a blind spot to hide). Use a combination of rotation and translation changes for the sensor, not only pure rotation on the spot for best possible results.
For each sensor the data (.png for images, or .npy for the point clouds) should be placed in a folder with its name (can be chosen freely but must match later the ones used in the config file! We simply used the ros topic names without the /).

For the recorded data a `calibration.yaml` config file needs to be created. Please put all the topic names for the cameras and LiDAR, as well as roughly measure the initial guess. Additional, define which cameras are standard perspective cameras ("pinhole") or fisheye cameras ("fisheye"), e.g.

```
camera_models:
- pinhole
- pinhole
- pinhole
- pinhole
- fisheye
- fisheye
```

If no information is given in the yaml file, perspective cameras are assumed.
An example can be of how the yaml should look like can be found in the provided test data.

We provide exemplary data here:
- Images, Point Clouds, configuration file: [Download here](https://www.ipb.uni-bonn.de/html/projects/ipb_calibration/calibration_data.zip)

### Compute Calibration

To run the calibration use `lcba.py` for calibrating the ipb-car. One can probably leave most of the default parameters as they are.
In the end the point clouds should be well aligned to the reference map and the camera rays should intersect the apriltag corners.

## Docker for system calibration

A configuration for docker is available. We provide a Dockerfile together with a docker compose configuration which mounts necessary directories
from the hosts system for input and output. To prepare a run in a docker container provide the following:

- **Reference data** in folder `reference`: 
  - provide the reference point cloud (see above) as file `reference/reference_pointcloud.ply`
  - provide the Apriltag coordinates (see above) as file `reference/apriltag_coords.txt`
- **Input data** in folder `input`: provide the data recording (see above) together with the `calibration.yaml` config file in folder `input`.

The docker image can be build and the calibration can be started with
(working directory should be root directory of the repository):
```
docker compose up --build
```
(To separate building and start use `docker build .` and `docker compose up`.)

After the calibration, the output is available in folder `output`:
- `result1.log`: Logfile for the console output of the calibration run
- `result1/results.yaml`: (also available as pickle file): The actual
  calibration result with extrinsics and intrinsics together
  with their a-posteriori covariance matrices of the cameras and lasers.
  `result1/args.yaml`: The settings used for the calibration. 

**Notes**:
- If you need to change some parameters of the calibration program, 
  modify the start script `entrypoint.bash`
- if you need visualization (e.g. using --visualize in `entrypoint.bash`),
  then allow X11 connections for user root with
```
xhost local:root
```
  before starting the container.
- The compose file assumes that a NVIDIA GPU is available. If no visualization is needed 
  (and that is the default !) and you have no nvidia GPU, the "deploy" section the
  `compose.yaml` can be commented out.


## Usage without docker

We highly recommend using docker. In case it is not needed you can take a look at the Dockerfile to see what needs to be installed. 

Most importantly:
Install the [AprilTag library](https://github.com/AprilRobotics/apriltag/releases/tag/v3.4.2) (tested under 3.4.2).
Install this repository (e.g. `pip install -e .`)
