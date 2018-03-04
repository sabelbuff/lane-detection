# laneDetection
This project implements various computer vision algorithms to detect and show left and right lanes in the given video generated using front-facing car mounted camera.

The video is converted into individual frames. Each frame is processed separately before being merged together to produce the output video containing detected lane line in red color.

Example frame extracted from the video:

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image.png)

# Pipeline to process each frame of the video is as follows:

- Firstly, the image was converted to the HSV colorspace.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image_hsv.png)

- Gaussian Blur was applied to the above image.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image_blur.png)

- To the image obtained from above, color filtering was applied, which masked the whole image other than the tints of yellow                 and white.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image_color_filtering.png)

- Now, the masked image was converted to grayscale.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image_grayscale.png)

- Canny detection algorithm was applied to the grayscale image output the edges in the image.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image-canny.png)

- Then region of interest was defined for the image because we are only interested in the lane part of the image.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image_roi.png)

- Hough transform was applied to this region of interest to extract the lane lines.
-----The draw_lines function used to draw lines on the image was modified to extract the points of the left and right lane --------separately based on the slope. 
-----Then, points were fitted by a line using linear regression(L2 loss). 
-----The slope and intercept that was outputted from the line fitting then averaged with the parameters of the previous frames -----to derive new slopes and intercepts. This ensures that the lane lines are smooth over the course of the video.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image_hough.png)

- The image from above is then merged to the original image to get the final result.

![alt text](https://raw.githubusercontent.com/sabelbuff/laneDetection/master/readme_images/image_lanes.png)


