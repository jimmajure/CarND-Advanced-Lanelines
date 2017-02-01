'''
Created on Jan 21, 2017

@author: jim
'''
import cv2
import glob
from utilities import imread
import numpy as np
import pickle

if __name__ == '__main__':
    # chessboard is 9x6
    nx = 9
    ny = 6
    
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('./camera_cal/*.jpg')
    
    # Step through the list and search for chessboard corners
    
    num_valid = 0
    for idx, fname in enumerate(images):
        img = imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
        # If found, add object points, image points
        if ret == True:
            num_valid += 1
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
    
    print("num valid: {}".format(num_valid))
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (720,1280), None, None)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/distortion_pickle.p", "wb" ) )

                        
                        