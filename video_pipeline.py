import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import pickle
from image_thresholding import *
from plotting_helpers import *

# Class Line
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.curv = None
        # distance in meters of vehicle center from the line
        self.base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # fitting method
        self.fit_method = ""
        # frames without updating
        self.skip_frames = 50
        self.err_msg = ""

    def update(self, y, fit, x, curv):
        self.detected = True
        self.current_fit = fit
        self.allx = x
        self.ally = y
        self.curv = curv
        self.base_pos = self.allx[len(self.ally)-1]


# Functions for polynomial fitting step
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    nwindows = 9        # Choose the number of sliding windows
    margin = 100        # Set the width of the windows +/- margin
    minpix = 50         # Set minimum number of pixels found to recenter window

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
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
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # the four boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = [i for i in range(len(nonzerox)) if
                          win_xleft_low < nonzerox[i] < win_xleft_high and
                          win_y_low < nonzeroy[i] < win_y_high]
        good_right_inds = [i for i in range(len(nonzerox)) if
                           win_xright_low < nonzerox[i] < win_xright_high and
                           win_y_low < nonzeroy[i] < win_y_high]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # Recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds).astype(int)
        right_lane_inds = np.concatenate(right_lane_inds).astype(int)
    except ValueError:      # Avoids an error if the above is not implemented fully
        print("Something bad happened")
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def find_lane_around_fit(binary_warped):
    # HYPERPARAMETER
    margin = 100
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    x_left = [left_lane.current_fit[0] * y ** 2 + left_lane.current_fit[1] * y + left_lane.current_fit[2]
              for y in range(binary_warped.shape[0])]
    x_right = [right_lane.current_fit[0] * y ** 2 + right_lane.current_fit[1] * y + right_lane.current_fit[2]
               for y in range(binary_warped.shape[0])]

    left_lane_inds = [i for i in range(len(nonzerox)) if
                      x_left[nonzeroy[i]] - margin < nonzerox[i] < x_left[nonzeroy[i]] + margin]
    right_lane_inds = [i for i in range(len(nonzerox)) if
                       x_right[nonzeroy[i]] - margin < nonzerox[i] < x_right[nonzeroy[i]] + margin]

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img


def sanity_chk(ploty, left_fit_cf, right_fit_cf, left_px, right_px, left_curv, right_curv):
    xm_per_pix = 3.7/700        # meters per pixel in x dimension
    sane = True
    err = ""
    width = (right_px[len(ploty)-1] - left_px[len(ploty)-1]) * xm_per_pix
    if width > 4.2 or width < 3.2:                      # Is lane width around 3.7m (+/- 0.5)?
        sane = False
        err = "No right lane distance"
    lane_dist = right_px - left_px
    lane_delta = lane_dist - np.mean(lane_dist)
    if np.max(lane_delta) > np.mean(lane_dist) / 10:    # Are lanes more or less parallel?
        sane = False
        err = err + " - " + "No parallel lanes"
    if left_curv < 2000 or right_curv < 2000:           # Is the curvature similar?
        if abs(left_curv - right_curv) > 200:
            sane = False
            err = err + " - " + "No same curvature"
    return sane, err


def find_curv(ally, left_fit, right_fit):
    ym_per_pix = 30 / 720       # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700      # meters per pixel in x dimension
    ally_m = ally * ym_per_pix
    def curv(fit):
        a = (xm_per_pix / ym_per_pix ** 2) * fit[0]
        b = (xm_per_pix / ym_per_pix) * fit[1]
        return np.mean(((1 + (2 * a * ally_m + b) ** 2) ** (3 / 2)) / abs(2 * a))
    return curv(left_fit), curv(right_fit)


def fit_and_update(leftx, lefty, rightx, righty, y_px):
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, y_px - 1, y_px)
    left_px = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_px = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # calculate curvature
    left_curv, right_curv = find_curv(ploty, left_fit, right_fit)
    # perform sanity check on new lines
    sane, err = sanity_chk(ploty, left_fit, right_fit, left_px, right_px, left_curv, right_curv)
    # if fine, we update the lanes values with this last fit
    if sane or left_lane.fit_method == "Boxes":
        left_lane.update(ploty, left_fit, left_px, left_curv)
        right_lane.update(ploty, right_fit, right_px, right_curv)
        left_lane.skip_frames = 0
        left_lane.err_msg = err
    # if wrong we don't update and go with previous fit for some frames until doing the box method
    else:
        left_lane.detected = False
        right_lane.detected = False
        left_lane.skip_frames += 1
        left_lane.err_msg = err


def lane_offset(screen_size):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    lane_w = (right_lane.base_pos - left_lane.base_pos) * xm_per_pix
    offset = (((right_lane.base_pos + left_lane.base_pos) - screen_size) / 2) * xm_per_pix
    return lane_w, offset


# *** PIPELINE ***
def pipeline(img):
    global error_im
    # Open distorsion matrix
    try:
        saved_dist = pickle.load(open('calibrate_camera.p', 'rb'), encoding='latin1')
        mtx = saved_dist['mtx']
        dist = saved_dist['dist']
    except (OSError, IOError):  # No progress file yet available
        print("No saved distorsion data. Run camera_calibration.py")

    # 1. Correct distorsion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # 2. Apply filters to get binary map
    ksize = 3
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(20, 100))
    dir_binary = dir_threshold(undist, sobel_kernel=15, thresh=(0.9, 1.1))
    hls_binary = hls_select(img, thresh=(90, 255))
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1 | ((mag_binary == 1) & (dir_binary == 1))) | hls_binary == 1] = 1
    # Plot the thresholding step
    # plot_thresholds(undist, gradx, grady, mag_binary, dir_binary, hls_binary, combined)

    # 3. Define trapezoid points on the road and transform perspective
    X = combined.shape[1]
    Y = combined.shape[0]
    src = np.float32(
            [[200, 720],
             [1100, 720],
             [700, 460],
             [580, 460]])
    dst = np.float32(
            [[300, 720],
             [980, 720],
             [980, 0],
             [300, 0]])
    # Get perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the result of binary thresholds
    warped = cv2.warpPerspective(combined, M, (X,Y), flags=cv2.INTER_LINEAR)
    # Plot warping step
    # plot_warping(combined, warped)

    # 4. Get polinomial fit of lines
    # If lanes were detected on previous frame search for new lane around that location
    if left_lane.skip_frames < 5:
        leftx, lefty, rightx, righty, out_img = find_lane_around_fit(warped)
        left_lane.fit_method = "Around fit"
        right_lane.fit_method = "Around fit"
        fit_and_update(leftx, lefty, rightx, righty, warped.shape[0])
    # if we already skipped some frames (or at the beginning, as we initialize frames to a high value)
    # then do boxes method
    else:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        left_lane.fit_method = "Boxes"
        right_lane.fit_method = "Boxes"
        fit_and_update(leftx, lefty, rightx, righty, warped.shape[0])
    # out_img = fit_polynomial(warped)
    # Plot polynomial result
    # plot_img(out_img)

    # 5. Calcutale curvature and distance to center
    lane_w, offset = lane_offset(img.shape[1])

    # 6. Plot fitted lanes into original image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.allx, left_lane.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.allx, right_lane.ally])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # if error save original img to check closely in image pipeline
    if left_lane.skip_frames == 0 and left_lane.err_msg != "":
        mpimg.imsave(left_lane.err_msg + "_" + str(error_im) + ".jpg", img)
        error_im += 1

    # Add text
    curv_left_txt = str(int(left_lane.curv))
    curv_right_txt = str(int(right_lane.curv))
    lane_width_txt = "%.2f" % lane_w
    lane_off_txt = "%.2f" % offset
    cv2.putText(result, "Curvature left lane: " + curv_left_txt + "m",
                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.putText(result, "Curvature right lane: " + curv_right_txt + "m",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(result, "Lane width: " + lane_width_txt + "m",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(result, "Offset: " + lane_off_txt + "m",
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(result, left_lane.fit_method,
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if left_lane.err_msg != "":
        cv2.putText(result, "Error!: " + left_lane.err_msg,
                    (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return result


clip_name = "project_video"
clip1 = VideoFileClip(clip_name + ".mp4")#.subclip(0, 14)

left_lane = Line()
right_lane = Line()
error_im = 1

out_clip = clip1.fl_image(pipeline)
out_clip.write_videofile(clip_name + "_output.mp4", audio=False)
