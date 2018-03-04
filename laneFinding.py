import os
import helperFunctions as hf
import numpy as np
from moviepy.editor import VideoFileClip
import globalVariables as gb


def process_image(image):
    """
    This Function process a frame from a video and draws lane lines over
    that image.
    """
    # change the colorspace of the image
    hsv_image = hf.rgb_to_hsv(image)

    # apply GaussianBlur
    kernel_size = 5
    image_blur = hf.gaussian_blur(hsv_image, kernel_size)

    # filter all colors of the image except the tints of yellow and white
    masked_image = hf.yellow_and_white_mask(image_blur)

    # convert the image to gray scale
    gray_image = hf.rgb_to_gray(masked_image)

    # apply canny transform on the gray scale image
    low_threshold = 50
    high_threshold = 150
    edges = hf.canny(gray_image, low_threshold, high_threshold)

    # filter the area of interest fro the image
    imshape = image.shape
    vertices = [(0, imshape[0]), (imshape[1] / 2 - 30, 2 * imshape[0] / 3 - 20),
                (imshape[1] / 2 + 30, 2 * imshape[0] / 3 - 20), (imshape[1], imshape[0])]
    roi = hf.region_of_interest(edges, np.array([vertices], dtype=np.int32))

    # extract hough lines from the image
    rho = 2
    theta = np.pi / 180
    threshold = 10
    min_line_length = 40
    max_line_gap = 20
    line_image = hf.hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)

    # add the hough lines to the original image
    final_image = hf.weighted_img(line_image, image)

    return final_image


output_folder = "test_videos_output/"
if not os.path.exists(output_folder): os.makedirs(output_folder)

gb.previous_frame_lines_left = []
gb.previous_frame_lines_right = []

# lane lines for right white lane
white_output = 'test_videos_output/solidWhiteRight.mp4'

clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

# lane lines for the left yellow lane
yellow_output = 'test_videos_output/solidYellowLeft.mp4'

clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

# Extended test video
final_output = 'test_videos_output/finalVideo.mp4'

clip3 = VideoFileClip('test_videos/finalVideo.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(final_output, audio=False)

# e.o.f
