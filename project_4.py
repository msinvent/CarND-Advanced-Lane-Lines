#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:38:20 2018

@author: manish
"""
import numpy as np
import glob, cv2
import imageio


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
        self.inverseM = None
        self.left_fit = None
        self.right_fit = None
        self.left_curverad = None
        self.right_curverad = None
        self.curvature = None
        
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
        self.inverseM = cv2.getPerspectiveTransform(dst,src)
        
    def __perspectiveTransformImage__(self,undistortedImage):
        return cv2.warpPerspective(undistortedImage, self.M, (self.imageShape[1], self.imageShape[0]))
    
    def __dewarpImage__(self,distortedImage):
        return cv2.warpPerspective(distortedImage, self.inverseM, (self.imageShape[1], self.imageShape[0]))
        
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
        
        fname = self.testImageLocation + "straight_lines1.jpg"
        img = cv2.imread(fname)
        undistortedImage = self.__undistortImage__(img)
#            gray = cv2.cvtColor(undistordedImage, cv2.COLOR_BGR2GRAY)
        offset = 400
        img_size = (img.shape[1], img.shape[0])
        
        src = np.array([[205,720],[596,450],[685,450],[1100,720]], np.double).reshape(4,1,2)
    
        dst = np.float32([
                [offset, img_size[1]],
                [offset, 0], 
                [img_size[0]-offset, 0],
                [img_size[0]-offset, img_size[1]] 
                ])

#        Don't delete these comments            
#        cv2.line(undistortedImage,(np.int(src[0].reshape(2,)[0]),np.int(src[0].reshape(2,)[1])),(np.int(src[1].reshape(2,)[0]),np.int(src[1].reshape(2,)[1])),(0,255,0),4)# left ( down to up)
#        cv2.line(undistortedImage,(np.int(src[2].reshape(2,)[0]),np.int(src[2].reshape(2,)[1])),(np.int(src[3].reshape(2,)[0]),np.int(src[3].reshape(2,)[1])),(0,255,0),4)# right (up to down)

        self.__updatePerspectiveTransormMatrix__(np.float32(src),np.float32(dst))
        warpedImage = self.__perspectiveTransformImage__(undistortedImage)
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
                
    def __ms_undistortImage__(self,img):
            if self.verbosity == 2:
                print('inside __perspectiveTranformOnRoadtest__ function')
            return self.__undistortImage__(img)            
    
    def __ms_perspectiveTranformOnRoadtest__(self,img):
            if self.verbosity == 2:
                print('inside __perspectiveTranformOnRoadtest__ function')
            return self.__perspectiveTransformImage__(img)            
                
    # Define a function that thresholds the S-channel of HLS
    # Use exclusive lower bound (>) and inclusive upper (<=)
    def __hls_select__(self,img, thresh=(90, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # 2) Apply a threshold to the S channel
        L = hls[:,:,1]
        S = hls[:,:,2]
        binary_output = np.zeros_like(S)
        
        # third (L > 100) is helping in removing shadows
        # L > 200 is helping in detecting white lines
        binary_output[((S > thresh[0]) & (S <= thresh[1]) & (L > 60)) | (L > 200)] = 255
        # 3) Return a binary image of threshold result
        return binary_output
    
    def __find_lane_pixels__(self,binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 17
        # Set the width of the windows +/- margin
        margin = 110
        # Set minimum number of pixels found to recenter window
        minpix = 30
    
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
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low =  leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
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
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        return leftx, lefty, rightx, righty, out_img

    def __fit_polynomial__(self,binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.__find_lane_pixels__(binary_warped)
    
        # Fit a second order polynomial to each using `np.polyfit` ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

    
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
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

        road_pnts = np.hstack((left_line,right_line))
        cv2.fillPoly(another_img,np.int_([road_pnts]),(0,255,0))
#        cv2.imwrite('./lane_boundaries.jpg',another_img)
    
        return another_img
    
    def __measure_curvature_real__(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
#        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Start by generating our fake example data
        # Make sure to feed in your real data instead in your project!
        left_fit_cr = self.left_fit
        right_fit_cr = self.right_fit
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum sey-value, corresponding to the bottom of the image
        y_eval = self.imageShape.shape[0]
        
        # Calculation of R_curve (radius of curvature)
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
         
    def __image_pipeline__(self):
        if self.verbosity == 2:
            print('inside __image_pipeline__ pipeline')
        self.__undistort__()
        self.__perspectiveTranformOnChessBoard__()
        self.__perspectiveTranformOnRoad__()
        self.__perspectiveTranformOnRoadtest__()

        imageNames = glob.glob(self.testImageLocation + "test*.jpg")  
        
        for fname in imageNames:
            img = cv2.imread(fname)
            img = self.__ms_undistortImage__(img)
#            cv2.imwrite(self.outputImageLocation + 'pipeline_002' + fname.split('/')[-1],img)
            # need to be done before perspective transform as the image is gettng blurry in the other case and it is difficult to identify edges and colors
            binary_threshold_image = self.__hls_select__(img)
#            cv2.imwrite(self.outputImageLocation  + 'pipeline_003' + fname.split('/')[-1],binary_threshold_image)
            warpedImage = self.__perspectiveTransformImage__(binary_threshold_image)
#            cv2.imwrite(self.outputImageLocation + 'pipeline_004' + fname.split('/')[-1],warpedImage)
            out_img = self.__fit_polynomial__(warpedImage)
#            cv2.imwrite(self.outputImageLocation + 'pipeline_005' + fname.split('/')[-1],out_img)
            self.__measure_curvature_real__()
            self.curvature = np.int_((self.left_curverad+self.right_curverad)/2)
            dewarpedImage = self.__dewarpImage__(out_img)
#            cv2.imwrite(self.outputImageLocation + 'pipeline_006' + fname.split('/')[-1],dewarpedImage)
#            print(np.zeros((self.imageShape[0],self.imageShape[1],1)).shape)
            lane_line = cv2.addWeighted(np.int_(img),1,np.int_(dewarpedImage),0.4,0)
            cv2.putText(lane_line,'Curvature = ' + str(self.curvature) + 'm' ,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            cv2.imwrite(self.outputImageLocation + 'pipeline_007' + fname.split('/')[-1],lane_line)


            
    def __video_pipeline__(self):
        if self.verbosity == 2:
            print('inside __video_pipeline__ pipeline')
        
        reader = imageio.get_reader('project_video.mp4','ffmpeg')
        
        for i, img in enumerate(reader):
#            print('Mean of frame %i is %1.1f' % (i, im.mean()))
            print('Processing frame : ',i)
            img = img[:,:,::-1]
            img = self.__ms_undistortImage__(img)
            binary_threshold_image = self.__hls_select__(img)
            warpedImage = self.__perspectiveTransformImage__(binary_threshold_image)
            out_img = self.__fit_polynomial__(warpedImage)
            self.__measure_curvature_real__()
            self.curvature = np.int_((self.left_curverad+self.right_curverad)/2)
            dewarpedImage = self.__dewarpImage__(out_img)
            lane_line = cv2.addWeighted(np.int_(img),1,np.int_(dewarpedImage),0.4,0)
            cv2.putText(lane_line,'Curvature = ' + str(self.curvature) + ' m' ,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            cv2.imwrite(self.outputImageLocation + 'videoframe_' + str(i) + '.jpg',lane_line)



#        while image in vid.get_data():
##            image = vid.get_data(num)
#            fig = pylab.figure()
#            fig.suptitle('image #{}'.format(1), fontsize=20)
#            pylab.imshow(image)
#        pylab.show()


def main():
    print("\033[2J")
    print('This is the main function that will take care of the pipeline of the image processing')
    params = {'calibrationImageLocation':'./camera_cal/',
              'outputImageLocation':'./output_images/',
              'testImageLocation':'./test_images/',
              'imageShape':np.array([720,1280,3]),
              'verbosity':2,
              'chessBoardDimension':(9,6)}

    createdObject = Line(objectName = 'Advanced Lane Detection Pipeline',**params)
    createdObject.__image_pipeline__();
    createdObject.__video_pipeline__();

if __name__ == "__main__":
    main()


