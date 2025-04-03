#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract apriltag coordinates from a scan of a room by 
creating orthophotos of the walls and detecting tags on the intensity of the laser.

Created on Thu Oct 17 16:43:33 2024

@author: laebe
"""
import os
import numpy as np
import argparse
import yaml
import open3d as o3d
from ipb_calibration.create_orthophoto_images import create_orthophoto_images, read_ptx_pointcloud
from ipb_calibration.detect_apriltags import detect_apriltags

def create_geometry_at_points(points, radius=0.005):
  """
    For 3D points create spheres as markers.
    Args:
      points:
        An nx3 array of 3D coordinates
      radius:
        raduis of the spheres.
    
    Returns:
      open3d geometry with list of spheres.
      
  """  
  geometries = o3d.geometry.TriangleMesh()
  for point in points:
      sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius) #create a small sphere to represent point
      sphere.translate(point) #translate this sphere to point
      geometries += sphere
  geometries.paint_uniform_color([1.0, 0.0, 0.0])
  return geometries



if __name__ == '__main__':

  #%% Settings
  parser = argparse.ArgumentParser(description='Extract apriltag coordinates from a scan of a room by '
                                   ' creating orthophotos of the walls and detecting tags on the intensity of the laser. ')
  parser.add_argument("yaml_config_file", type=str,
                      help="yaml configuration file for the walls in the room")
  parser.add_argument('--visualize', '-v', action='store_true', help =
                      'show all the extracted tags on the pointcloud.')
  
  args = parser.parse_args()
  
  yaml_config_file = args.yaml_config_file

  with open(yaml_config_file, 'r') as file:
    config = yaml.safe_load(file)
    
  # Default args
  if 'max_pyramid_level' in config:
    max_pyramid_level = config['max_pyramid_level']
  else:
    max_pyramid_level = 2
    
  # Loop over all walls  
  for wallname in config['walls'].keys():
    # Step 1: Create orthophoto
    odir = os.path.join(config['working_dir'], wallname)
    print('\n---------------------------------------------------------------')
    print('Creating orthophoto in %s ...' % odir, flush=True)
    create_orthophoto_images(
      config['scanfilename'], odir, config['pixel_size'], config['walls'][wallname]['bounding_box'],
      config['walls'][wallname]['projection_direction'])
    
    # Step 2: Apriltag extraction on that orthophoto
    detect_apriltags(odir)

  # Combine all extracted coordinates to one file
  print('\n Combine all coordinates ...', flush=True)
  coords = np.zeros((0,6))

  for wallname in config['walls'].keys():
    odir = os.path.join(config['working_dir'], wallname)
    coords_one_wall = np.loadtxt(os.path.join(odir, 'apriltag_coords_rc.txt'), skiprows=1)
    if len(coords_one_wall)>0:
      coords = np.concatenate((coords, coords_one_wall))
  
  print('Save ...', flush=True)
  # as text file with ID X Y Z 
  np.savetxt(os.path.join(config['working_dir'], 'apriltag_coords.txt'), coords[:, :4], fmt='%4d   %10.5f %10.5f %10.5f',
             header='%d' % coords.shape[0], comments='')
    
  print('\n... done with everthing.') 
  
  # Visualization
  if args.visualize:
    # Load
    print("Visualization ... ", flush=True)
    _, file_extension = os.path.splitext(config['scanfilename'])
    if file_extension == ".ptx":
      pcd = read_ptx_pointcloud(config['scanfilename'])
    else:
      pcd = o3d.io.read_point_cloud(config['scanfilename'])
       
    point_markers = create_geometry_at_points(coords[:,1:4],radius=0.01)
    
    o3d.visualization.draw_geometries([pcd, point_markers])
    
    
