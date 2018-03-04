import numpy as np
import cv2
import globalVariables as gb
# import math


def rgb_to_gray(img):
    """
    Applies the Grayscale transform
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(img):
    """
    convert the image to hsv colorspace
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def yellow_and_white_mask(img):
    """
    mask the image wherever there is no tint of yellow and white
    because lane lines are either yellow or white
    """
    min_yellow = np.array([20, 80, 80], dtype=np.uint8)
    max_yellow = np.array([30, 255, 255], dtype=np.uint8)
    min_white = np.array([0, 0, 200], dtype=np.uint8)
    max_white = np.array([179, 30, 255], dtype=np.uint8)

    mask_yellow = cv2.inRange(img, min_yellow, max_yellow)
    mask_white = cv2.inRange(img, min_white, max_white)

    yellow_or_white_mask = cv2.bitwise_or(mask_white, mask_yellow)
    return cv2.bitwise_and(img, img, mask=yellow_or_white_mask)


def canny(img, low_threshold, high_threshold):
    """
    Applies the Canny transform
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    """
    return cv2.GaussianBlur(img, [kernel_size, kernel_size], 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def cal_slope_intercept(vx, vy, x, y):
    """
    calculate the slope and intercept of the line
    """
    slope = vy / vx
    intercept = y - slope * x
    return slope, intercept


def cal_endpoints_x(avg, start_y, end_y):
    """
    calculate the x co-ordinate of the end points of the line
    """
    intercept = avg[1]
    slope = avg[0]
    start_x = int((start_y - intercept) / slope)
    end_x = int((end_y - intercept)/ slope)
    return start_x, end_x


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function separates line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, it takes average of the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function also draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    """
    # signifies left and right lane lines
    left_line = np.empty((0, 2))
    right_line = np.empty((0, 2))

    imshape = img.shape

    # contains slope and intercept from previous frames of the video to smooth the lane lines
    # global previous_frame_lines_left, previous_frame_lines_right

    start_y = img.shape[0]
    end_y = int(2 * imshape[0] / 3 - 20)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # arrange the points according to left or right lane based on their slopes
            if (y2 - y1) / (x2 - x1) < 0.5:
                left_line = np.append(left_line, np.array([[x1, y1]]), axis=0)
                left_line = np.append(left_line, np.array([[x2, y2]]), axis=0)
            elif (y2 - y1) / (x2 - x1) > 0.5:
                right_line = np.append(right_line, np.array([[x1, y1]]), axis=0)
                right_line = np.append(right_line, np.array([[x2, y2]]), axis=0)

    if left_line.size:
        # fit a line through the points
        [left_line_vx, left_line_vy, left_line_x, left_line_y] = cv2.fitLine(np.array(left_line, dtype=np.int32),
                                                                             cv2.DIST_L2, 0, 0.01, 0.01)

        [slope_left, intercept_left] = cal_slope_intercept(left_line_vx, left_line_vy, left_line_x, left_line_y)

        # add the slope snd intercept of this frame to previous frames and take average for smoother lane line
        gb.previous_frame_lines_left.append([slope_left[0], intercept_left[0]])
        average = np.mean(gb.previous_frame_lines_left, axis=0)

        [start_left_line_x, end_left_line_x] = cal_endpoints_x(average, start_y, end_y)

        # draw the line on the image through the end points
        cv2.line(img, (start_left_line_x, start_y), (end_left_line_x, end_y), color, thickness)

    elif not gb.previous_frame_lines_left == []:
        # if the frame has no lines, use slope and intecept from previous frames for continuity
        average = np.mean(gb.previous_frame_lines_left, axis=0)

        [start_left_line_x, end_left_line_x] = cal_endpoints_x(average, start_y, end_y)

        cv2.line(img, (start_left_line_x, start_y), (end_left_line_x, end_y), color, thickness)

    if right_line.size:
        # fit a line through the points
        [right_line_vx, right_line_vy, right_line_x, right_line_y] = cv2.fitLine(np.array(right_line, dtype=np.int32),
                                                                                 cv2.DIST_L2, 0, 0.01, 0.01)

        [slope_right, intercept_right] = cal_slope_intercept(right_line_vx, right_line_vy, right_line_x, right_line_y)

        # add the slope snd intercept of this frame to previous frames and take average for smoother lane line
        gb.previous_frame_lines_right.append([slope_right[0], intercept_right[0]])
        average = np.mean(gb.previous_frame_lines_right, axis=0)

        [start_right_line_x, end_right_line_x] = cal_endpoints_x(average, start_y, end_y)

        # draw the line on the image through the end points
        cv2.line(img, (start_right_line_x, start_y), (end_right_line_x, end_y), color, thickness)

    elif not gb.previous_frame_lines_right == []:
        # if the frame has no lines, use slope and intercept from previous frames for continuity
        average = np.mean(gb.previous_frame_lines_right, axis=0)

        [start_right_line_x, end_right_line_x] = cal_endpoints_x(average, start_y, end_y)

        cv2.line(img, (start_right_line_x, start_y), (end_right_line_x, end_y), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=5)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

