import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import pickle
from image_thresholding import *
from plotting_helpers import *
from line_fit import *

# Class Line
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # number of frames to keep in history
        self.nframes = 4
        # x values of the last n fits of the line
        self.fit_x = None
        self.fit_x_hist = []
        # y values for detected line pixels
        self.fit_y = None
        # polynomial coefficients averaged over the last n iterationss
        self.fit_hist = []
        self.fit_avg = None
        # radius of curvature of the line in some units
        self.curv_avg = None
        self.curv_hist = []
        # distance in meters of vehicle center from the line
        self.base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

    def update(self, y, fit, x, curv):
        self.fit_y = y
        if len(self.fit_x_hist) >= self.nframes:
            self.fit_x_hist.pop(0)
            self.fit_hist.pop(0)
            self.curv_hist.pop(0)
        self.fit_x_hist.append(x)
        self.fit_hist.append(fit)
        self.curv_hist.append(curv)

        self.fit_x = np.mean(self.fit_x_hist, axis=0)
        self.fit_avg = np.mean(self.fit_hist, axis=0)
        self.curv_avg = np.mean(self.curv_hist)

        self.base_pos = self.fit_x[len(self.fit_y)-1]


def find_curv(ally, left_fit, right_fit):
    ym_per_pix = 30 / 720       # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700      # meters per pixel in x dimension
    ally_m = ally * ym_per_pix
    def curv(fit):
        a = (xm_per_pix / ym_per_pix ** 2) * fit[0]
        b = (xm_per_pix / ym_per_pix) * fit[1]
        return np.mean(((1 + (2 * a * ally_m + b) ** 2) ** (3 / 2)) / abs(2 * a))
    return curv(left_fit), curv(right_fit)


def lane_offset(screen_size):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    lane_w = (right_lane.base_pos - left_lane.base_pos) * xm_per_pix
    offset = (((right_lane.base_pos + left_lane.base_pos) - screen_size) / 2) * xm_per_pix
    return lane_w, offset


# *** PIPELINE ***
def pipeline(img):
    global error_im, skipped_frames, fit_method

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
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(5, 100))
    mag_bin = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(10, 200))
    dir_bin = dir_threshold(undist, sobel_kernel=15, thresh=(0.9, 1.2))
    hls_bin = hls_select(img, thresh=(100, 255))
    white_bin = white_select(img, thresh=175)
    yellow_bin = yellow_select(img)
    combined = np.zeros_like(dir_bin)

    combined[((mag_bin == 1) & ((dir_bin == 1) & (hls_bin == 1) & (white_bin == 1) & (yellow_bin == 1))) |
             ((white_bin == 1) | (yellow_bin == 1))] = 1

    # 3. Define trapezoid points on the road and transform perspective
    X = combined.shape[1]
    Y = combined.shape[0]
    src = np.float32(
            [[205, 720],
             [1075, 720],
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

    # 4. Get polinomial fit of lines
    # if > 4 frames skipped (or first frame, as skipped_frames is initialized to 100) do full search
    if skipped_frames > 4:
        fit_method = "Boxes"
        # find pixels
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        # fit polynomials
        left_fit, right_fit, left_px, right_px, ploty = fit(leftx, lefty, rightx, righty, warped.shape[0])
        # sanity check
        ok, err_msg = sanity_chk(ploty, left_px, right_px)
        if ok:
            skipped_frames = 0
        else:
            skipped_frames += 1
        left_curv, right_curv = find_curv(ploty, left_fit, right_fit)
        left_lane.update(ploty, left_fit, left_px, left_curv)
        right_lane.update(ploty, right_fit, right_px, right_curv)
    else:           # If lanes were detected on previous frame search for new lane around that location
        fit_method = "Around fit"
        # find pixels
        leftx, lefty, rightx, righty, out_img = find_lane_around_fit(warped, left_lane.fit_x, right_lane.fit_x)
        # fit polynomials
        left_fit, right_fit, left_px, right_px, ploty = fit(leftx, lefty, rightx, righty, warped.shape[0])
        # sanity check
        ok, err_msg = sanity_chk(ploty, left_px, right_px)
        if ok:
            skipped_frames = 0
            left_curv, right_curv = find_curv(ploty, left_fit, right_fit)
            left_lane.update(ploty, left_fit, left_px, left_curv)
            right_lane.update(ploty, right_fit, right_px, right_curv)
        else:
            skipped_frames += 1

    # 5. Calcutale curvature and distance to center
    lane_w, offset = lane_offset(img.shape[1])

    # 6. Plot fitted lanes into original image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.fit_x, left_lane.fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.fit_x, right_lane.fit_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # if error save original img to check closely in image pipeline
    if 1 < skipped_frames < 3:
        mpimg.imsave(err_msg + "_" + str(error_im) + ".jpg", img)
        error_im += 1

    # Add text
    if left_lane.curv_avg > 2000:
        curv_left_txt = "straight"
    else:
        curv_left_txt = str(int(left_lane.curv_avg)) + "m"
    if right_lane.curv_avg > 2000:
        curv_right_txt = "straight"
    else:
        curv_right_txt = str(int(right_lane.curv_avg)) + "m"
    lane_width_txt = "%.2f" % lane_w
    lane_off_txt = "%.2f" % offset
    cv2.putText(result, "Curvature left lane: " + curv_left_txt,
                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.putText(result, "Curvature right lane: " + curv_right_txt,
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(result, "Lane width: " + lane_width_txt + "m",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(result, "Offset: " + lane_off_txt + "m",
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(result, fit_method,
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if err_msg != "":
        cv2.putText(result, "Error!: " + err_msg,
                    (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return result


clip_name = "challenge_video"
clip1 = VideoFileClip(clip_name + ".mp4").subclip(0, 8)

left_lane = Line()
right_lane = Line()
error_im = 1
skipped_frames = 100
fit_method = ""

out_clip = clip1.fl_image(pipeline)
out_clip.write_videofile(clip_name + "_output.mp4", audio=False)
