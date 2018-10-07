#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:38:20 2018

@author: manish
"""




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


def main():
    print('This is the main function that will take care of the pipeline of the image processing')
    
    

if __name__ == "__main__":
    main()


