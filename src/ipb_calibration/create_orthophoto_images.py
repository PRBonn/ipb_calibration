#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an orthophoto-type projection of a pointcloud. Used for:
Extracting apriltags from Faro laser scanner data.

Created on Wed Jul 21 15:30:07 2021

@author: laebe
"""
import argparse
import os
from pathlib import Path
import open3d as o3d
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import tqdm

# Faster load if already loaded once
#from diskcache import FanoutCache
#import getpass
#cache = FanoutCache(
#    directory=os.path.join("/tmp", "fanoutcache_" + getpass.getuser() + "/"),
#    shards=64,
#    timeout=1,
#    size_limit=3e11,
#)

def create_orthophoto_images(pointcloud_filename: str, odir:str, pixelsize: float, bb:list, 
                             projection_direction: int=3, use_as_column_axis:int = -1,
                             use_as_row_axis:int = -1, visualize:bool = False):
  """
    Create orthophoto-like intensity,x,y,z coordinate image from a (Faro laserscanner) pointcloud.
  
    Args:  
      pointcloud_filename:
        filename for the pointcloud. User either .ptx for 
        the original file from Faro or any other type possible
        for open3d read function.
  
      odir: 
        output directory. Will be created if not there.
        Written files are
        intensity.tif  xcoord.tif  ycoord.tif  zcoord.tif.
                                                
      pixelsize:
        "GSD or Size of a pixel in object coordinates (usually meter).
  
      bb:
        Bounding box. The pointcloud is cropped with this bounding box.
        Give 6 values as a list: minX, minY, minZ, maxX, maxY, maxZ.
                    
      projection_direction:
        X axis is 1, Y axis 2, Z axis 3. This is
        the coordinate which is set to 0 to get the orthographic projection.
        If you look into negative axis direction, give -1 for X, -2 for Y,
        -3 for Z. Note that looking into positive/negative direction means
        that the orthophoto is flipped/mirrored. Then the tag code
        cannot be recognized any more.
      
      use_as_column_axis:
        Columm axis of the orthophoto.X axis is 0, Y axis 1, Z axis 2. If not
        given, as suitable default is used. Note that not giving that axis,
        the orthophoto might be flipped. Not given is a value <0.
  
      use_as_row_axis:
        Row axis of the orthophoto.X axis is 0, Y axis 1, Z axis 2. If not
        given, as suitable default is used. Note that not giving that axis
        the orthophoto might be flipped. Not given is a value <0.
      
      visualize:
        show the orthophoto (the intensity of the orthophoto).
  """
  # Load
  print("Load pointcloud %s ... " % pointcloud_filename, flush=True)
  _, file_extension = os.path.splitext(pointcloud_filename)
  if file_extension == ".ptx":
    pcd = read_ptx_pointcloud(pointcloud_filename)
    #print("Write ply ...",flush=True)
    #o3d.io.write_point_cloud('/tmp/t.ply', pcd, write_ascii=False, compressed=True)
  else:
    pcd = o3d.io.read_point_cloud(pointcloud_filename)
  

  # crop out 
  min_bound=np.array([bb[0], bb[1], bb[2]]).T
  max_bound=np.array([bb[3], bb[4], bb[5]]).T
  bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
  pcd = pcd.crop(bbox)

  # pcd in 2D format
  X = np.asarray(pcd.points)
  X2D = X.copy()
  X2D[:, np.abs(projection_direction)-1]=0
  pcd2D = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X2D))
  intensity = np.asarray(pcd.colors).copy()
  intensity = 255.0 * intensity[:,0]
  
  # define the axis of the orthophoto
  # This are the default axis:
  if np.abs(projection_direction)==1:
    first_orthindex = 1
    second_orthindex = 2
  else:    
    if np.abs(projection_direction)==2:
      first_orthindex = 2
      second_orthindex = 0
    else:
      # Default,should be projection along Z
      first_orthindex = 0
      second_orthindex = 1
      
  # If projection direction is negative, we look into negative direction,
  # thus the orthophoto has to be mirrored. Do this by swapping r/c axis
  # of orthophoto
  if projection_direction<0:
    h = first_orthindex
    first_orthindex = second_orthindex
    second_orthindex = h
    
      
  # If axis given by the user, use them
  if use_as_column_axis>=0:
    first_orthindex = use_as_column_axis
  if use_as_row_axis>=0:
    second_orthindex= use_as_row_axis
  
  # show
  # o3d.visualization.draw_geometries([pcd])

  # define orthophoto
  min_bound = pcd.get_min_bound()
  max_bound = pcd.get_max_bound()
  min_firstidx = min_bound[first_orthindex]
  min_secondidx = min_bound[second_orthindex]
  # X == columns, Y = rows
  no_rows = int(np.ceil( (max_bound[second_orthindex] - min_bound[second_orthindex]) / pixelsize))
  no_cols = int(np.ceil( (max_bound[first_orthindex] - min_bound[first_orthindex]) / pixelsize))
  print("Resulting images have size (rows x cols): %d x %d " % (no_rows, no_cols), flush=True)
  
  img_intensity = np.zeros((no_rows, no_cols), dtype='uint8')
  img_coord = np.zeros((no_rows, no_cols,3), dtype='float32')
  
  # Search tree
  print("Create search tree ...", flush=True)
  kdtree = o3d.geometry.KDTreeFlann(pcd2D)
  
  # And go
  print("Create images ...", flush=True)
  query = np.zeros((3,1), dtype='float32')
  orthoidxs = [first_orthindex, second_orthindex]
  for r in tqdm.tqdm(range(no_rows)):
    #print("row %d of %d ..." % (r, no_rows), flush=True)
    for c in range(no_cols):
      query[first_orthindex] = min_firstidx + pixelsize * c
      query[second_orthindex] = min_secondidx + pixelsize * r
      _, idx, _ = kdtree.search_knn_vector_3d(query, 4)
      radii = np.linalg.norm(X[idx,:][:, orthoidxs] - query[orthoidxs].T, axis=1)
      weights = radii / np.sum(radii)
      img_intensity[r,c] = np.round(np.sum(weights * intensity[idx]))
      img_coord[r,c,:] = np.sum(np.expand_dims(weights, axis=1) * X[idx,:], axis=0)
  
  # Save
  Path(odir).mkdir(parents=True, exist_ok=True)
  tifffile.imsave(os.path.join(odir, 'intensity.tif'), np.expand_dims(img_intensity, axis=2))
  tifffile.imsave(os.path.join(odir, 'xcoord.tif'), img_coord[:,:,0:1])
  tifffile.imsave(os.path.join(odir, 'ycoord.tif'), img_coord[:,:,1:2])
  tifffile.imsave(os.path.join(odir, 'zcoord.tif'), img_coord[:,:,2:3])
  
  #show
  if visualize:
    plt.figure(1);
    plt.clf();
    plt.imshow(img_intensity, cmap='gray', vmin=0, vmax=255)
    plt.pause(0.02)
    plt.show() 
  

#@cache.memoize(typed=True)
def read_ptx_pointcloud(filename: str) -> o3d.geometry.PointCloud:
  """ 
  Read a pointcloud in .ptx format. This format is from the Faro laser scanner.
  It should be ASCII and has x,y,i,r,g,b in every line (and an addtional header)
  
  Args:
    filename: the complete filename of the file
  
  Returns:
    an open3d pointcloud which has the intensity as color (all values the same for rgb)
  """
  data = np.loadtxt(filename, skiprows=10, dtype='float32')
  
  pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[:,0:3]))
  pcd.colors = o3d.utility.Vector3dVector(np.stack([data[:,3], data[:,3], data[:,3]]).T)
  return pcd



if __name__ == "__main__":
  """
  Create an orthophoto-type projection of a pointcloud. Used for:
  Extracting apriltags from Faro laser scanner data.
  """

  #%% Parse arguments
  parser = argparse.ArgumentParser(description="create orthophoto-like intensity,x,y,z coordinate image from a (Faro laserscanner) pointcloud")
  parser.add_argument("pointcloud_filename", help=("filename for the pointcloud. User either .ptx for "
                                                  "the original file from Faro or any other type possible "
                                                  "for open3d read function."))
  parser.add_argument("--output_directory", "-o", help=("default output directory. Will be created "
                                                        "if not there. Written files are "
                                                "intensity.tif  xcoord.tif  ycoord.tif  zcoord.tif. "
                                                "Default is ./"), default="./")    
  parser.add_argument("--pixelsize", "-s", default=0.001, type=float,
                      help="GSD or Size of a pixel in object coordinates (usually meter). Default: 0.001")
  parser.add_argument("--bounding_box", "-b",  nargs=6, type=float,
                      help="Bounding box. The pointcloud is cropped with this bounding box. "
                           "Give 6 values: minX, minY, minZ, maxX, maxY, maxZ "
                           "default= -10.0 -10.0 -10.0  10.0 10.0 10.0]",
                           default=[-10.0, -10.0, -10.0, 10.0, 10.0, 10.0])
  parser.add_argument("--projection_direction", default=2, type=int,
                      help="Projection direction: X axis is 1, Y axis 2, Z axis 3. This is"
        "the coordinate which is set to 0 to get the orthographic projection. "
        "If you look into negative axis direction, give -1 for X, -2 for Y, "
        "-3 for Z. Note that looking into positive/negative direction means "
        "that the orthophoto is flipped/mirrored. Then the tag code "
        "cannot be recognized any more. Default: 3.")
  parser.add_argument('--visualize', '-v', action='store_true', help =
                      'show the orthophoto (the intensity of the orthophoto).')

  # Settings
  args = parser.parse_args()
  pointcloud_filename = args.pointcloud_filename
  odir = args.output_directory
  pixelsize = args.pixelsize
  bb = args.bounding_box
  projection_direction = args.projection_direction
  use_as_column_axis = args.use_as_column_axis
  use_as_row_axis = args.use_as_row_axis
  visualize = args.visualize
  
  create_orthophoto_images(pointcloud_filename, odir, pixelsize, bb, 
                           projection_direction, visualize=visualize)
  
