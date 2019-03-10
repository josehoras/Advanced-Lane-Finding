import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from image_thresholding import *
from plotting_helpers import *
from line_fit import *


def find_curvature(left_fit_cf, right_fit_cf, ploty):
    # Calculate curvature of fits
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    r_y = ploty * ym_per_pix

    def curv(fit_cf):
        a = (xm_per_pix / ym_per_pix ** 2) * fit_cf[0]
        b = (xm_per_pix / ym_per_pix) * fit_cf[1]
        return np.mean(((1 + (2 * a * r_y + b)**2)**(3/2)) / abs(2 * a))
    return curv(left_fit_cf), curv(right_fit_cf)


# *** PIPELINE ***

# Get image
img_name = " - No parallel lanes_4.jpg"
# img_name = "test_images/straight_lines2.jpg"
img = mpimg.imread(img_name)
print(img.shape)

# 1. Correct distorsion
# Open distorsion matrix
try:
    saved_dist = pickle.load(open('calibrate_camera.p', 'rb'), encoding='latin1')
    mtx = saved_dist['mtx']
    dist = saved_dist['dist']
except (OSError, IOError):  # No progress file yet available
    print("No saved distorsion data. Run camera_calibration.py")
# get undistorted image
undist = cv2.undistort(img, mtx, dist, None, mtx)
# plot_calibration(img, undist)

# 2. Apply filters to get binary map
ksize = 3
gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(10, 100))
grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(5, 100))
mag_bin = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(10, 200))
dir_bin = dir_threshold(undist, sobel_kernel=15, thresh=(0.75, 1.25))
hls_bin = hls_select(img, thresh=(80, 255))
white_bin = white_select(img, thresh=175)
yellow_bin = yellow_select(img)
combined = np.zeros_like(dir_bin)

combined[((mag_bin == 1) & (dir_bin == 1)) & ((hls_bin == 1) | (white_bin == 1) | (yellow_bin == 1))] = 1

# Plot the thresholding step
plot_thresholds(undist, mag_bin, dir_bin,
                hls_bin, white_bin, yellow_bin,
                ((mag_bin == 1) & (dir_bin == 1)), combined,
                ((hls_bin == 1) | (white_bin == 1) | (yellow_bin == 1)))

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
# Plot warping step
# plot_warping(combined, warped, src)

# 4. Get polinomial fit of lines
plot_poly = True
out_img, left_fit_cf, right_fit_cf,  left_fitx, right_fitx, ploty = fit_polynomial(warped, plot=plot_poly)
# leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
ok, err = sanity_chk(ploty, left_fitx, right_fitx)
print(ok, err)
# Plot polynomial result
if plot_poly: plot_img(out_img)

# 5. Calcutale curvature
curv_left, curv_right = find_curvature(left_fit_cf, right_fit_cf, ploty)

print("base dist:  ", right_fitx[len(ploty)-1] - left_fitx[len(ploty)-1])
print("upper dist: ", right_fitx[0] - left_fitx[0])
print("Curvature left: ", curv_left)
print("Curvature right: ", curv_right)




# 6. Plot fitted lanes into original image
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
# print("dst: ", dst.shape)
# print("newwarp: ", newwarp.shape)
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plot_img(result)
