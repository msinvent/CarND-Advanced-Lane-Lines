#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:38:20 2018

@author: manish
"""
import numpy as np
import glob, cv2
import matplotlib.pyplot as plt



#    Project Steps 
#    Camera calibration 
#    Distortion correction 
#    Color/gradient threshold 
#    Perspective transform 
#    Detect lane lines 
#    Determine the lane curvature 

# Finding camera calibration matrix
def findWarpingCoef(src,dst):
    M = None
    return M

# Testing the output of the image warping on a test image
def warper(img, src, dst):

    # Compute and apply perpective transform
#    img_size = (img.shape[1], img.shape[0])
#    M = cv2.getPerspectiveTransform(src, dst)
#    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    warped = None
    return warped

def imageThresholding(image):
    warpedThresholdedImage = None
    return warpedThresholdedImage;

def perspectiveTrasform(image):
    T = None
    returnImage = image*T
    return returnImage

 # Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, imageShape = np.array([720,1280,3]) ,nx=9, ny =6, objectName = 'defaultName', calibrationImageLocation = './',outputImageLocation = './', verbosity = 0,chessBoardDimension = (9,6),  **params):
        # set the shape of input image
        self.imageShape = imageShape
        # specify an object name in case we create multiple objects
        self.objectName = objectName
        # specify the location of camera calibration image
        self.calibrationImageLocation = calibrationImageLocation
        # specify the location of output images
        self.outputImageLocation = outputImageLocation
        # specify if the verbosity of the print statements
        self.verbosity = verbosity
        # specify the chessboard dimension
        self.chessBoardDimension = chessBoardDimension
        # the number of inside corners in x
        self.nx = self.chessBoardDimension[0]
        # the number of inside corners in y
        self.ny = self.chessBoardDimension[1] 
        # Camera calibration matrix
        self.cameraMatrix = None
        self.distortionParameters = None
        
        if self.verbosity >= 1:    
            print('Initialzation of the Object "',objectName,'" which is of type "Line" is complete')
                
    def __undistort__(self):
        # Ref : Udacity lectures and https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        if self.verbosity == 2:
            print('inside undistort function')
        
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.nx*self.ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        imageNames = glob.glob(self.calibrationImageLocation + "calibration*.jpg")
        
        globalRet = False
        for fname in imageNames:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)
            
            if ret == True:
                globalRet = True
                objpoints.append(objp)
#                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                corners2 = corners
                imgpoints.append(corners2)
            else:
                print('failed to detect chess board corners for ', fname)
            
        # If we have atleast one image with corners detected        
        if globalRet == True:
            ret, self.cameraMatrix, self.distortionParameters, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            h,  w = self.imageShape[:2]

        else:
            raise ValueError('Not able to calculate the distortion error')
        
#        for fname in imageNames:
#            img = cv2.imread(fname)
#            # undistort
#            dst = cv2.undistort(img, self.cameraMatrix, self.distortionParameters, None, self.cameraMatrix)
##            cv2.imshow('UndistortedImage',dst)
#            break
         
    def __perspectiveTransform__(self):
        if self.verbosity == 2:
            print('inside perspective Transform function')
            
        imageNames = glob.glob(self.calibrationImageLocation + "calibration*.jpg")
        
        for fname in imageNames:
            img = cv2.imread(fname)
            undistordedImage = cv2.undistort(img, self.cameraMatrix, self.distortionParameters, None, self.cameraMatrix)
            gray = cv2.cvtColor(undistordedImage, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            
            if ret == True:
                # If we found corners, draw them! (just for fun)
                cv2.drawChessboardCorners(undistordedImage, (self.nx, self.ny), corners, ret)
                offset = 100 # offset for dst points
                # Grab the image shape
                img_size = (gray.shape[1], gray.shape[0])
                
                # For source points I'm grabbing the outer four detected corners
                src = np.float32([corners[0], corners[self.nx-1], corners[-1], corners[-self.nx]])

                dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
                # Given src and dst points, calculate the perspective transform matrix
                M = cv2.getPerspectiveTransform(src, dst)
                # Warp the image using OpenCV warpPerspective()
                warped = cv2.warpPerspective(undistordedImage, M, img_size)
                cv2.imwrite(self.outputImageLocation + 'perspectiveTransformOutput_' + fname.split('/')[-1],warped)
#                cv2.imshow(fname,warped)
#                cv2.waitKey()
                
                
                
    def __image_pipeline__(self):
        if self.verbosity == 2:
            print('inside image pipeline')
        self.__undistort__()
        self.__perspectiveTransform__()
#
#        
#        fname = 'calibration_test.png'
#        img = cv2.imread(fname)
#
#        # Convert to grayscale
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#        # Find the chessboard corners
#        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#
#        # If found, draw corners
#        if ret == True:
#            # Draw and display the corners
#            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
#            plt.imshow(img)

#    def __corners_unwarp__(img, mtx, dist):
#        # Use the OpenCV undistort() function to remove distortion
#        undist = cv2.undistort(img, mtx, dist, None, mtx)
#        # Convert undistorted image to grayscale
#        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
#        # Search for corners in the grayscaled image
#        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#    
#        if ret == True:
#            # If we found corners, draw them! (just for fun)
#            cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
#            # Choose offset from image corners to plot detected corners
#            # This should be chosen to present the result at the proper aspect ratio
#            # My choice of 100 pixels is not exact, but close enough for our purpose here
#            offset = 100 # offset for dst points
#            # Grab the image shape
#            img_size = (gray.shape[1], gray.shape[0])
#    
#            # For source points I'm grabbing the outer four detected corners
#            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#            # For destination points, I'm arbitrarily choosing some points to be
#            # a nice fit for displaying our warped result 
#            # again, not exact, but close enough for our purposes
#            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
#                                         [img_size[0]-offset, img_size[1]-offset], 
#                                         [offset, img_size[1]-offset]])
#            # Given src and dst points, calculate the perspective transform matrix
#            M = cv2.getPerspectiveTransform(src, dst)
#            # Warp the image using OpenCV warpPerspective()
#            warped = cv2.warpPerspective(undist, M, img_size)
#
#        # Return the resulting image and matrix
#        return warped, M


def main():
    print("\033[2J")
    print('This is the main function that will take care of the pipeline of the image processing')
    params = {'calibrationImageLocation':'../camera_cal/',
              'outputImageLocation':'../output_images/',
              'imageShape':np.array([720,1280,3]),
              'verbosity':2,
              'chessBoardDimension':(9,6)}

    createdObject = Line(objectName = 'Advanced Lane Detection Pipeline',**params)
    createdObject.__image_pipeline__();

if __name__ == "__main__":
    main()


