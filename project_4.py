#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:38:20 2018

@author: Manish Sharma

#    Project Steps : 
#    Camera calibration 
#    Distortion correction 
#    Color/gradient threshold 
#    Perspective transform 
#    Detect lane lines 
#    Determine the lane curvature 
"""

import numpy as np
import glob, cv2
import imageio

# Define a class to receive the characteristics of each line detection
class laneDetection():
    # laneDetection pipeline constructor
    def __init__(self, imageShape = np.array([720,1280,3]) ,nx=9, ny =6, objectName = 'defaultName', 
                 calibrationImageLocation = './',outputImageLocation = './', testImageLocation = './',
                 verbosity = 0,chessBoardDimension = (9,6),
                 nwindows = 17, margin = 110, minpix = 30, **params):
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
        self.inverseM = None
        self.left_fit = None
        self.right_fit = None
        self.left_curverad = None
        self.right_curverad = None
        self.curvature = None
        self.distanceFromCenter = None

        # Set number of row windows to divide the image        
        self.nwindows = nwindows
        # Set the width of the windows +/- margin
        self.margin = margin
        # Set minimum number of pixels found to recenter window
        self.minpix = minpix
        
        
        ## Calculate initial cameraMatrix and distortionParameters
        self.__undistort__()
        self.__perspectiveTranformOnChessBoard__()
        self.__calculatePerspectiveTranformMatrixFromRoadSample__()
        self.__perspectiveTranformOnRoadtest__()
        
        if self.verbosity >= 1:    
            print('Initialzation of the Object "',objectName,'" which is of type "Line" is complete')

    # function to calculate camera matrix using distortion parameters               
    def __undistort__(self):
        # Ref : Udacity lectures and https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        if self.verbosity == 2:
            print('inside __undistort__ function')
        
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
        
         
    # This function take input image and returns undistorted image as output using camera matrix and distortion parameters calculated using
    def __undistortImage__(self,img):
        if self.verbosity == 2:
            print('inside __undistortImage__ function')
        return cv2.undistort(img, self.cameraMatrix, self.distortionParameters, None, self.cameraMatrix)

    # Update projective transform matrix this need to be done once for the whole video as we are assuming flat groud model
    def __updateCameraMatrix__(self,src,dst):
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.inverseM = cv2.getPerspectiveTransform(dst,src)
        
    # warp image to see it from the top view(front view) to calculate curvature
    def __warpImage__(self,undistortedImage):
        return cv2.warpPerspective(undistortedImage, self.M, (self.imageShape[1], self.imageShape[0]))
    
    # take the image from top view to original frame, used to reconstruct the green lane over the top of original image
    def __dewarpImage__(self,distortedImage):
        return cv2.warpPerspective(distortedImage, self.inverseM, (self.imageShape[1], self.imageShape[0]))
        
    # uses different chessboards available to find the perspective transform
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
    
                # Given src and dst points, calculate the perspective transform matrix
                self.__updateCameraMatrix__(src,dst)
                # Warp the image using OpenCV warpPerspective()
                warpedImage = self.__warpImage__(undistordedImage)
                cv2.imwrite(self.outputImageLocation + 'warped_' + fname.split('/')[-1],warpedImage)
            else:
                print('not able to find automatic chessboard corners thus not able to perform image warping, saving only undistorted image')
                cv2.imwrite(self.outputImageLocation + 'undistorted_' + fname.split('/')[-1],undistordedImage)
        
        
    def __calculatePerspectiveTranformMatrixFromRoadSample__(self):
        if self.verbosity == 2:
            print('inside __perspectiveTranformOnRoad__ function')
        
        fname = self.testImageLocation + "straight_lines1.jpg"
        img = cv2.imread(fname)
        undistortedImage = self.__undistortImage__(img)
        offset = 300
        img_size = (img.shape[1], img.shape[0])
        
        src = np.array([[205,720],[596,450],[685,450],[1100,720]], np.double).reshape(4,1,2)
    
        dst = np.float32([
                [offset, img_size[1]],
                [offset, 0], 
                [img_size[0]-offset, 0],
                [img_size[0]-offset, img_size[1]] 
                ])

#        Don't delete these            
        cv2.line(undistortedImage,(np.int(src[0].reshape(2,)[0]),np.int(src[0].reshape(2,)[1])),(np.int(src[1].reshape(2,)[0]),np.int(src[1].reshape(2,)[1])),(0,255,0),4)# left ( down to up)
        cv2.line(undistortedImage,(np.int(src[2].reshape(2,)[0]),np.int(src[2].reshape(2,)[1])),(np.int(src[3].reshape(2,)[0]),np.int(src[3].reshape(2,)[1])),(0,255,0),4)# right (up to down)

        self.__updateCameraMatrix__(np.float32(src),np.float32(dst))
        warpedImage = self.__warpImage__(undistortedImage)
        cv2.imwrite(self.outputImageLocation + 'afterPerspectiveTransform_' + fname.split('/')[-1],warpedImage)
    
    def __perspectiveTranformOnRoadtest__(self):
            if self.verbosity == 2:
                print('inside __perspectiveTranformOnRoadtest__ function')
            
            imageNames = glob.glob(self.testImageLocation + "test*.jpg")    
            for fname in imageNames:
                img = cv2.imread(fname)
                undistordedImage = self.__undistortImage__(img)
                undistordedImage = img
                warpedImage = self.__warpImage__(undistordedImage)            
                cv2.imwrite(self.outputImageLocation + 'afterPerspectiveTransform_' + fname.split('/')[-1],warpedImage)
                
    def __ms_undistortImage__(self,img):
            if self.verbosity == 2:
                print('inside __perspectiveTranformOnRoadtest__ function')
            return self.__undistortImage__(img)            
    
    def __ms_perspectiveTranformOnRoadtest__(self,img):
            if self.verbosity == 2:
                print('inside __perspectiveTranformOnRoadtest__ function')
            return self.__warpImage__(img)            
                
    # Define a function that thresholds the S-channel of HLS
    # Use exclusive lower bound (>) and inclusive upper (<=)
    def __hls_select__(self,img, thresh=(90, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # 2) Apply a threshold to the S channel
        L = hls[:,:,1]
        S = hls[:,:,2]
        binary_output = np.zeros_like(S)
        
        # third (L > 60) is helping in removing shadows
        # L > 200 is helping in detecting white lines
        binary_output[((S > thresh[0]) & (S <= thresh[1]) & (L > 60)) | (L > 200)] = 255
        # 3) Return a binary image of threshold result
        return binary_output
    
    
    # takes a binary image and return leftx, lefty, rightx, righty
    def __find_lane_pixels__(self,binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = self.nwindows
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix
    
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low =  leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
        
            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high) & 
            (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox>=win_xright_low) & (nonzerox<win_xright_high) &
            (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append( good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            self.lastValid_leftx = nonzerox[left_lane_inds]
            self.lastValid_lefty = nonzeroy[left_lane_inds] 
            self.lastValid_rightx = nonzerox[right_lane_inds]
            self.lastValid_righty = nonzeroy[right_lane_inds]
        except ValueError:
            print('error in concatenating lane pixels using previous valid indexes')
            return self.lastValid_leftx, self.lastValid_lefty, self.lastValid_rightx, self.lastValid_righty
    
        return leftx, lefty, rightx, righty


    # Finds the polynomial fitting to the left and right lane corners and also the distance from the center of the lane in metre
    def __fit_polynomial__(self,binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty = self.__find_lane_pixels__(binary_warped)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            # Fit a second order polynomial to each using `np.polyfit` ###
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        another_img = np.zeros((binary_warped.shape[0],binary_warped.shape[1],3))

        left_line = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
        right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
        
        leftLanePosition = left_fitx[self.imageShape[0]-1]
        rightLanePosition = right_fitx[self.imageShape[0]-1]
        leftLanePosition = np.array([leftLanePosition,719,1])
        rightLanePosition = np.array([rightLanePosition,719,1])

        # finding the lane edges start point in the original frame to find the exact distance from center
        leftLanePositionOriginal = np.matmul(self.inverseM,leftLanePosition) 
        rightLanePositionOriginal = np.matmul(self.inverseM,rightLanePosition) 
        leftLanePositionOriginal = leftLanePositionOriginal/leftLanePositionOriginal[2]
        rightLanePositionOriginal = rightLanePositionOriginal/rightLanePositionOriginal[2]
        
        laneCenterPosition = (leftLanePositionOriginal[0] + rightLanePositionOriginal[0])/2
        ImageCenterPosition = self.imageShape[1]/2
        
        mx = 3.7/700 # meters per pixel in x dimension
        self.distanceFromCenter = (ImageCenterPosition - laneCenterPosition)*mx     
        
        road_pnts = np.hstack((left_line,right_line))
        cv2.fillPoly(another_img,np.int_([road_pnts]),(0,255,0))
        return another_img
    
    def __measure_curvature_real__(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        my = 30/720 # meters per pixel in y dimension
              
        # left and right lane boundaries coefficient in hy
        left_fit_cr = self.left_fit 
        right_fit_cr = self.right_fit
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum sey-value, corresponding to the bottom of the image
        y_eval = self.imageShape[0] #720
        
        # Calculation of R_curve (radius of curvature in metres)
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*my + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*my + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
         
#        print(y_eval)
        
    def __pipeline__(self,img):
        img = self.__ms_undistortImage__(img)
        binary_threshold_image = self.__hls_select__(img)
        cv2.imwrite('binarythreshold.jpg',binary_threshold_image)
        warpedImage = self.__warpImage__(binary_threshold_image)
        cv2.imwrite('warpedImage.jpg',warpedImage)
        out_img = self.__fit_polynomial__(warpedImage)
        cv2.imwrite('fitPolynomial.jpg',out_img)
        self.__measure_curvature_real__()
        # approximation of curvature of center of the lane using the curvature of left lane boundary and right lane boundary
        self.curvature = np.int_((self.left_curverad+self.right_curverad)/2)
        dewarpedImage = self.__dewarpImage__(out_img)
        imageWithLane = cv2.addWeighted(np.int_(img),1,np.int_(dewarpedImage),0.4,0)
        cv2.putText(imageWithLane,'Curvature = ' + str(self.curvature) + ' m' ,(150,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        if(self.distanceFromCenter < 0):
            cv2.putText(imageWithLane,'Car is driving ' + str(np.abs(self.distanceFromCenter)) + ' m' + ' left of lane center',(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        else:
            cv2.putText(imageWithLane,'Car is driving ' + str(np.abs(self.distanceFromCenter)) + ' m' + ' right of lane center',(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        return imageWithLane
        
    def __image_pipeline__(self):
        if self.verbosity == 2:
            print('inside __image_pipeline__ pipeline')

        imageNames = glob.glob(self.testImageLocation + "test*.jpg")  
        
        for fname in imageNames:
            img = cv2.imread(fname)
            imageWithLine = self.__pipeline__(img)
            cv2.imwrite(self.outputImageLocation + 'pipeline_007' + fname.split('/')[-1],imageWithLine)
#           
    def __video_pipeline__(self):
        if self.verbosity == 2:
            print('inside __video_pipeline__ pipeline')
        
        reader = imageio.get_reader('./project_video.mp4','ffmpeg')
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer('./output_images/project_video_output.mp4',fps=fps)
        for i, img in enumerate(reader):
            if self.verbosity >= 1 :
                print('Processing frame : ',i)
            imageWithLine = self.__pipeline__(img)
            writer.append_data(imageWithLine)
            # For debugging purposes, if you want to cut the processing short to some specific number of video frames
#            if i == 100:
#                break
            
        writer.close()

def main():
    print("\033[2J")
    print('This is the main function that will take care of the pipeline of the image processing')
    params = {'calibrationImageLocation':'./camera_cal/',
              'outputImageLocation':'./output_images/',
              'testImageLocation':'./test_images/',
              'imageShape':np.array([720,1280,3]),
              'verbosity':1,
              'chessBoardDimension':(9,6)}

    # Create the object of laneDetection class with the above parameters
    createdObject = laneDetection(objectName = 'Advanced Lane Detection Pipeline',**params)
    createdObject.__image_pipeline__();
    createdObject.__video_pipeline__();

if __name__ == "__main__":
    main()


