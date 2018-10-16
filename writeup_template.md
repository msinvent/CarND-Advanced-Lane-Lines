## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image2]: ./output_images/perspectiveTransformOutput_un_calibration1.jpg "Undistorted"
[image3]: ./output_images/binarythreshold.jpg "Binary Threshold"
[image4b]: ./output_images/perspectiveTransformOutput_calibration3.jpg "Another Warpeed Image"
[image4]: ./output_images/warpedImage.jpg "Warpeed Image"
[image5]: ./output_images/fitPolynomial.jpg "Fit Polynomial"
[image6]: ./output_images/pipeline_007test2.jpg "Lanes Transformed Back on the image"
[video1]: ./output_images/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

It is best to explain the final pipeline and explain it step by step :

The following image explains very much what I am doing to each individual image, I am starting with finding an undistortedImage and then doing a binary threshod, after which I am unwarping the image, after which I did lane pixel detection using section wise pixel detection.
```
 def __pipeline__(self,img):
        img = self.__ms_undistortImage__(img)
        binary_threshold_image = self.__hls_select__(img)
        warpedImage = self.__warpImage__(binary_threshold_image)
        out_img = self.__fit_polynomial__(warpedImage)
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
```

### Pipeline (single images)

#### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

![Distorted Image][image1]
I am showing this sectino with respect to the calibration1.jpg image given as an example insidet the camaera_cal folder with the project.

The methods __undistort__ is responsible for calculating the cameraMatrix and distortionParameters and I am using cv2.calibrateCamera function to feed in object points and final expected points, for which I am using cv2.findChessboardCorners function.
After running the __undistortImage__ on the image calibration1.jpg I am getting final result as the next image
![UnDistorted Image][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

```
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) 
	#Apply a threshold to the S channel
	L = hls[:,:,1]
	S = hls[:,:,2]
	binary_output = np.zeros_like(S)        
	#third (L > 60) is helping in removing shadows
	#L > 200 is helping in detecting white lines
	binary_output[((S > thresh[0]) & (S <= thresh[1]) & (L > 60)) | (L > 200)] = 255
```
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform matrix is calculated using the function 

```python
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
        self.__updateCameraMatrix__(np.float32(src),np.float32(dst))
        warpedImage = self.__warpImage__(undistortedImage)
        cv2.imwrite(self.outputImageLocation + 'afterPerspectiveTransform_' + fname.split('/')[-1],warpedImage)
```


![alt text][image4]

Another image showing the image warping is
![alt text][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used density based approach to find the max density portion in the frame portions to locate the pixels corresponding to the left and right lane, this is done in the function __find_lane_pixels__, after that I have used __fit_polynomial__ function to fit a 2nd order polynomial on left and right lanes, This is followed by converting the result from pixel space to meter space and then to calculate the final curvature of the lane I am simply averaging the results of the left and the right lanes.

* I am avoiding any fancy averaging for now because this code was looking sufficient enought to generate good lane lines, 
* have placed one check though that If the lane line is not detected then I am choosing the previous lane line as the current, the same idea can be extended to using the average of last few lane lines to calculate the present lane curvatures.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The measure curvature real is calculating the curvature in metres from the pixel space using 
```
self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*my + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*my + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final implementation can be seen in the next image. The lane lines are transformed back to the image space and then plotted on top of original image to show the final result, this is done using function __dewarpImage__ and then merging the result with the original image and adding the text on top of it.
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Well I will say that this code is very far from going into a real car, specially any change in the lighting condition is breaking it, may be an adaptive thresholding technique should be used to extend it to a more ggeneral scenario(even to pass the challenge video case)