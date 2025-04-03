# Apriltag extraction on TLS laser scans

We attached A4 sized apriltag of family 36h11 to the walls and then scanned the room.
The 3D scan of the room needs to contain for each point the 3D coordinates and the measured intensity (this is what you get with e.g. a Faro TLS scanner). We
use a ply file as input. (A ASCII ptx-file which can be exported in Faro Scene should also work.)

The procedure has two steps:
- produce an orthophoto using the pointcloud. This produces an intensity image as well as 
  float images with the corresponding X,Y,Z coordinates 
- apply the april tag detector on the intensity image and write out the corresponding x,y,z coordinates.

One can directly use a python script (see Usage) or use a docker container (see Docker).



## Usage

The whole process can be executed by a single script:

```
extract_all_apriltag_coords_on_scan.py [--visualize] <filename of yaml configuration file>
```

The room will be separated into walls and for each wall an orthophoto will be generated and the
apriltag detector will be called. Finally all detected 3D coordinates will be combined to
a single output file. If you use the `--visualize` the final detected coordinates will be shown
together with the pointcloud.

Everything is controlled by the yaml-File. An example can be found at
`config/extract_all_apriltag_coords.yaml`:
- define the name of the TLS laserscan with tag `scanfilename`. We assume that
  the walls (+ ceiling and maybe even floor) are roughly axis aligned, e.g. the
  left wall is parallel to the X-Z plane. If the coordinate origin is roughly in
  the middle of the room, it is more easy to separate the walls. In the given
  example configuration file we assume this.
- The final combined output file (filename `apriltag_coords.txt`)
  as well as all orthophotos images will be written
  to the directory `working_dir` (will be created if not present).
  Additional for every wall there will be two output files with the apriltag coordinates:
  `apriltag_coords.txt` and `apriltag_coords_rc.txt` which additionally includes
  the row and column coordinates of the detected tags on the intensity image. (center of
  upper left pixel is (0,0)).
- `pixel_size` defines the resolution/ground sampling distance of the orthophotos. For
  best results, this should be similar to the average nearest points distance of the
  pointcloud. Note that this parameter highly influence the computation time. You
  might choose a lower resolution for a first test and then use a high resolution for
  the final extraction.
- For every wall with apriltags you should define a configuration under the
  tag `wall` and give the following two attributes:
  - `projection direction`: Give the axis which is perpendicular to the wall. X=1, Y=2, Z=3
     If you look into negative axis direction (seen from the center), give -1 for X, -2 for Y, -3 for Z. 
     Note that looking into positive/negative direction means
     that the orthophoto is flipped/mirrored. Then the tag code
     cannot be recognized any more. Thus if you have a wall with april tags and no tags are found,
     maybe flip the projection direction.
  - `bounding box`: This is necessary to separate parallel walls, otherwise both walls will be
     projected on the same orthophoto. If there are no other objects in the room, this can
     be done very roughly, e.g. just filtering out all positive X coordinates. 
     If there are more objects that could obscure individual April tags, then the bounding
     box has to be defined more closer to the wall in order to detect all tags.


## Docker

A configuration for docker is available. We provide a Dockerfile `apriltag_extraction.Dockerfile` together with a docker compose configuration which mounts necessary directories
from the hosts system for input and output. To prepare a run in a docker container provide the following:

- **Input data** in folder `input`: provide the tls laser scan as file  `input/tls_scan.ply`.
- **Configuration** The yaml file `config/extract_all_apriltag_coords.yaml` is used for defining the walls of the room.
  You may need to modify this configuration (see Usage for a description)

The docker image can be build and the extraction can be started with
(working directory should be root directory of the repository):
```
docker compose --profile apriltag_extraction up --build
```

You will then find the results in the directory **`output/apriltag_extraction`**:
- the extracted coordinates in the file `apriltag_coords.txt`. For later usage for the calibration copy this file to folder
  `reference`
- a log of the extraction in `result.log`
