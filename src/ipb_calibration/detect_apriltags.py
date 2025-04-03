#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect apriltags in orthophotos.

Created on Thu Sep 26 15:07:51 2024

@author: laebe
"""
import os
import argparse
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import apriltag


def detect_apriltags(working_dir:str, max_pyramid_level:int = 2,
                                   visualize_tags:bool = False) ->np.ndarray:
  """
    Given a set of intensity and coordinates images (of an orthophoto), extract apriltag coordinates
    on these images. 
    
    The results are written into the working directory with files
    apriltagcoords.txt (containing ID, X Y Z coordinates) and
    apriltagcoords_rc.txt (containing ID, X Y Z row column coordinates).
    
    Args:
      working_dir:
        directory of the input orthophoto files 
        (intensity.tif, xcoord.tif, ycoord.tif, zcoord.tif). The output will be written 
        into this directory as file apriltagcoords.txt
        
     max_pyramid_level:
       Pyramid level of the smallest images used. In order to detect very large
       apriltags, the tag detection in not only done on the original image, but also 
       on higher image pyramid levels. This is the highest image pyramid level number. 
       '0 == original size, 1 == half rows/cols of original, ... Default is: 2
       
     visualize_tags:
       plot the extracted tags in a figure on the intensity image.
       Default: False
       
    Returns:
      all_tags_3D, thus a nx6 numpy array of all coordinates with
      columns ID X Y Z row column
      X,Y,Z: 3D coordinates of the extracted apriltag corners
      row, column: The intensity image row column coordinates. They are relative to the upper left pixel CENTER,
      thus the upper left corner of the upper left pixel has coordinates (-0.5, -0.5)
      
  """
  
  #%% Load images
  print('Load images ...', flush=True)
  image = tifffile.imread(os.path.join(working_dir, 'intensity.tif'))

  Ximage = tifffile.imread(os.path.join(working_dir, 'xcoord.tif'))
  Yimage = tifffile.imread(os.path.join(working_dir, 'ycoord.tif'))
  Zimage = tifffile.imread(os.path.join(working_dir, 'zcoord.tif'))
  
  #%% Detect
  detector = apriltag.apriltag("tag36h11", decimate=1, maxhamming=2)
  
  all_tags_2D = dict() # Tags with ID as keys, [r,c] as values
  
  
  for level in range(max_pyramid_level, -1, -1):
  #for level in (2,2):
    print('\nWorking on pyramid level %d ...' % level, flush=True)
    level_factor = 2 ** level
    if image.ndim == 3:
      img_level = np.copy(image[:,:,0])
    else:
      img_level = np.copy(image[:,:])
      
    for i in range(level):
      img_level = cv2.pyrDown(img_level)
    
    # call the actual detector
    results = detector.detect(img_level)
    
    # Convert from apriltag detector output to list with ID, c,r
    no_tags = len(results)
    for i in range(no_tags):
      #Our ID scheme: 100* (apriltag_id) + [1,2,3,4] for the four corners
      ids  = np.arange(100*results[i]['id'] + 1, 100*results[i]['id'] + 5)
      corners = results[i]['lb-rb-rt-lt']
      # Use our sequence for the numbering: upper left, upper right , bottom left, bottom right
      corners = corners[[3, 2, 0, 1]]
      
      # April tags seem to have coordinates where (0,0) is not the center, but the
      # upper left corner of a pixel. We want integer coordinates at the pixel center.
      # Thus substract half a pixel.
      corners = level_factor * (corners - 0.5)
      
      # Put into dictionary. Note that previous detections on higher pyramid levels will be overwritten
      for j in range(len(ids)):
        all_tags_2D[ ids[j] ] = corners[j, :]
        
    print('... detected %d tags on level %d.' % ( no_tags, level ) )
    
  print('\n... detected %d tags on all levels.' % ( len(all_tags_2D)/4 ) )
  
  #%% Extract 3D coordinates
  print('Extract 3D coordinates ...', flush=True)
  # Put into array with ID X Y Z r c
  all_tags_3D = np.zeros((len(all_tags_2D), 6))
  for tag_id, i in zip(all_tags_2D.keys(), range(len(all_tags_2D)) ):
    all_tags_3D[i, 0  ] = tag_id
    all_tags_3D[i, 4] = all_tags_2D[tag_id][1]
    all_tags_3D[i, 5] = all_tags_2D[tag_id][0]
    
    all_tags_3D[i, 1] = getInterpolated(Ximage, all_tags_3D[i, 4:6])
    all_tags_3D[i, 2] = getInterpolated(Yimage, all_tags_3D[i, 4:6])
    all_tags_3D[i, 3] = getInterpolated(Zimage, all_tags_3D[i, 4:6])
    
  #%% Save 
  print('Save ...', flush=True)
  # as text file with ID X Y Z 
  np.savetxt(os.path.join(working_dir, 'apriltag_coords.txt'), all_tags_3D[:, :4], fmt='%4d   %10.5f %10.5f %10.5f',
             header='%d' % all_tags_3D.shape[0], comments='')
    
  # as text file with ID X Y Z r c (r,c: (0,0) is upper left pixel center)
  np.savetxt(os.path.join(working_dir, 'apriltag_coords_rc.txt'), all_tags_3D, fmt='%4d   %10.5f %10.5f %10.5f  %8.2f %8.2f',
             header='%d' % all_tags_3D.shape[0], comments='')
  
  print('... done.') 
  
  #%% Plot
  if visualize_tags:
    plt.figure(1)
    plt.clf()
    plt.imshow(image, cmap='gray')
    plt.plot(all_tags_3D[:,5], all_tags_3D[:,4], 'r.') # plot -> c,r integers are at pixel centers
    plt.title('Extracted apriltag coordinates')
    plt.show()

  return all_tags_3D
  
  

def getInterpolated(img: np.ndarray, rc: np.ndarray) -> float:
  """ 
    Helper function: Bilinear interpolation on an image
    Args:
      img: 
        a 2D image
      rc: an array/list with two elements [row, colum]
      
  """
  rint = int(rc[0])
  cint = int(rc[1])

  a = rc[0] - rint
  b = rc[1] - cint

  return ( (img[rint, cint]     * (1.0 - a) + img[rint + 1, cint]     * a) * (1.0 - b) +
           (img[rint, cint + 1] * (1.0 - a) + img[rint + 1, cint + 1] * a) * b  )



if __name__ == '__main__':

  #%% Settings
  parser = argparse.ArgumentParser(description='Extract apriltag coordinates in an orthophoto.'
                                   ' An orthophoto in this context is a set of four images: '
                                   'an intensity image and 3 float images for the corresponding X,Y,Z coordinates '
                                   'of the pointcloud which was used for generating the orthophoto.')
  parser.add_argument('working_dir', help = 'directory of the input orthophoto files '
                      '(intensity.tif, xcoord.tif, ycoord.tif, zcoord.tif). The output will be written '
                      'into this directory as file apriltagcoords.txt')
  parser.add_argument('--max_pyramid_level', '-p', type = int, default = 2, help =
                      'Pyramid level of the smallest images used. In order to detect very large '
                      'apriltags, the tag detection in not only done on the original image, but also '
                      'on higher image pyramid levels. This is the highest image pyramid level number. '
                      '0 == original size, 1 == half rows/cols of original, ... Default is: 2')
  parser.add_argument('--visualize_tags', '-v', action='store_true', help =
                      'plot the extracted tags on the image.')
  
  args = parser.parse_args()
  working_dir = args.working_dir
  max_pyramid_level = args.max_pyramid_level
  visualize_tags = args.visualize_tags
  
  detect_apriltags(working_dir, max_pyramid_level, visualize_tags)
  
    
    