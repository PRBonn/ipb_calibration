#!/bin/bash
# Do the calibration
# This is just a call to a python script:

python3 calib/scripts/lcba.py --faro_file reference/reference_pointcloud.ply --apriltag_file reference/apriltag_coords.txt --path_data input --path_out output --experiment_name result1 | tee output/result1.log



