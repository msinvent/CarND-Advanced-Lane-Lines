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

def imageThresholding(image):
    warpedThresholdedImage = None
    return warpedThresholdedImage;

def perspectiveTrasform(image):
    T = None
    returnImage = image*T
    return returnImage

 # Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, imageShape = np.array([720,1280,3]) ,nx=9, ny =6, objectName = 'defaultName', 
                 calibrationImageLocation = './',outputImageLocation = './', testImageLocation = './',
                 verbosity = 0,chessBoardDimension = (9,6),  **params):
        # set the shape of input image
        self.imageShape = imageShape
        # specify an object name in case we create multiple objects
        self.objectName = objectName
        # specify the location of camera calibration image
        self.calibrationImageLocation = calibrationImageLocation
        # specify the location of output images
        self.outputImageLocation = outputImageLocation
        # specify the location of output images
        self.testImageLocation = testImageLocation
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
        self.M = None
        
        if self.verbosity >= 1:    
            print('Initialzation of the Object "',objectName,'" which is of type "Line" is complete')

# function to calculate self.cameraMatrix and self.distortionParameters               
    def __undistort__(self):
        # Ref : Udacity lectures and https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        if self.verbosity == 2:
            print('inside __undistort__ function')
        
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
        
         
    def __undistortImage__(self,img):
        if self.verbosity == 2:
            print('inside __undistortImage__ function')
        return cv2.undistort(img, self.cameraMatrix, self.distortionParameters, None, self.cameraMatrix)

            
    def __updatePerspectiveTransormMatrix__(self,src,dst):
        self.M = cv2.getPerspectiveTransform(src, dst)
        
    def __perspectiveTransformImage__(self,undistortedImage):
        return cv2.warpPerspective(undistortedImage, self.M, (self.imageShape[1], self.imageShape[0]))
        
    def __perspectiveTranformOnChessBoard__(self):
        if self.verbosity == 2:
            print('inside __perspectiveTranformOnChessBoard__ function')
            
        imageNames = glob.glob(self.calibrationImageLocation + "calibration*.jpg")
        
        for fname in imageNames:
            img = cv2.imread(fname)
            undistordedImage = self.__undistortImage__(img)
            gray = cv2.cvtColor(undistordedImage, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            
            if ret == True:
                # If we found corners, draw them! (just for fun)
                cv2.drawChessboardCorners(undistordedImage, (self.nx, self.ny), corners, ret)
                offset = 200 # offset for dst points
                # Grab the image shape
                img_size = (gray.shape[1], gray.shape[0])
                
                # For source points I'm grabbing the outer four detected corners
                src = np.float32([corners[0], corners[self.nx-1], corners[-1], corners[-self.nx]])

                dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
    
#                print(src.shape,'\n',dst.shape)
                # Given src and dst points, calculate the perspective transform matrix
#                M = cv2.getPerspectiveTransform(src, dst)
                self.__updatePerspectiveTransormMatrix__(src,dst)
                # Warp the image using OpenCV warpPerspective()
                warpedImage = self.__perspectiveTransformImage__(undistordedImage)
                cv2.imwrite(self.outputImageLocation + 'perspectiveTransformOutput_' + fname.split('/')[-1],warpedImage)
            else:
                cv2.imwrite(self.outputImageLocation + 'perspectiveTransformOutput_un_' + fname.split('/')[-1],undistordedImage)
        
    def __perspectiveTranformOnRoad__(self):
        if self.verbosity == 2:
            print('inside __perspectiveTranformOnRoad__ function')
        
        imageNames = glob.glob(self.testImageLocation + "straight_lines*.jpg")

#        (280,670)(1020,670)(585,460)(706,460)
        for fname in imageNames:
            img = cv2.imread(fname)
            undistordedImage = self.__undistortImage__(img)
#            gray = cv2.cvtColor(undistordedImage, cv2.COLOR_BGR2GRAY)
            offset = 200
            img_size = (img.shape[1], img.shape[0])
            
            src = np.array([[280,670],[525,500],[765,500],[1024,670]], np.double).reshape(4,1,2)
            dst = np.float32([[img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset], [offset, offset]])

            dst = np.float32([
                    [offset, img_size[1]-offset],
                    [offset, offset], 
                    [img_size[0]-offset, offset],
                    [img_size[0]-offset, img_size[1]-offset] 
                    ])

#            Don't delete these comments            
#            cv2.line(undistordedImage,(280,670),(580,465),(0,255,0),4)# left ( down to up)
#            cv2.line(undistordedImage,(710,470),(1024,670),(0,255,0),4)# right (up to down)
    
            self.__updatePerspectiveTransormMatrix__(np.float32(src),np.float32(dst))
            warpedImage = self.__perspectiveTransformImage__(undistordedImage)            
            cv2.imwrite(self.outputImageLocation + 'afterPerspectiveTransform_' + fname.split('/')[-1],warpedImage)
    
    def __perspectiveTranformOnRoadtest__(self):
            if self.verbosity == 2:
                print('inside __perspectiveTranformOnRoadtest__ function')
            
            imageNames = glob.glob(self.testImageLocation + "test*.jpg")    
            for fname in imageNames:
                img = cv2.imread(fname)
                undistordedImage = self.__undistortImage__(img)
                undistordedImage = img
                warpedImage = self.__perspectiveTransformImage__(undistordedImage)            
                cv2.imwrite(self.outputImageLocation + 'afterPerspectiveTransform_' + fname.split('/')[-1],warpedImage)
                
#    def __ms_perspectiveTranformOnRoadtest__(self,img):
#            if self.verbosity == 2:
#                print('inside __perspectiveTranformOnRoadtest__ function')
#            
#            imageNames = glob.glob(self.testImageLocation + "test*.jpg")    
#            for fname in imageNames:
#                img = cv2.imread(fname)
#                undistordedImage = self.__undistortImage__(img)
#                undistordedImage = img
#                warpedImage = self.__perspectiveTransformImage__(undistordedImage)            
#                cv2.imwrite(self.outputImageLocation + 'afterPerspectiveTransform_' + fname.split('/')[-1],warpedImage)
                
    # Define a function that thresholds the S-channel of HLS
    # Use exclusive lower bound (>) and inclusive upper (<=)
#    def __hls_select__(img, thresh=(0, 255)):
#        # 1) Convert to HLS color space
#        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#        # 2) Apply a threshold to the S channel
#        S = hls[:,:,2]
#        binary_output = np.zeros_like(S)
#        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
#        # 3) Return a binary image of threshold result
#        return binary_output
#    
            
    def __image_pipeline__(self):
        if self.verbosity == 2:
            print('inside image pipeline')
        self.__undistort__()
        self.__perspectiveTranformOnChessBoard__()
        self.__perspectiveTranformOnRoad__()
        self.__perspectiveTranformOnRoadtest__()
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
              'testImageLocation':'../test_images/',
              'imageShape':np.array([720,1280,3]),
              'verbosity':2,
              'chessBoardDimension':(9,6)}

    createdObject = Line(objectName = 'Advanced Lane Detection Pipeline',**params)
    createdObject.__image_pipeline__();

if __name__ == "__main__":
    main()


