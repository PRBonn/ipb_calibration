#!/bin/bash
# Do the apriltag extraction
# This is just a call to a python script:

python3 calib/scripts/extract_all_apriltag_coords_on_scan.py config/extract_all_apriltag_coords.yaml  | tee output/apriltag_extraction/apriltag_extraction_result.log



