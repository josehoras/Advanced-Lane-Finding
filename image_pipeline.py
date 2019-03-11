import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from image_thresholding import *
from plotting_helpers import *
from line_fit import *


# *** PIPELINE ***
# Get image
# img_name = " - No parallel lanes (1.5)_2.jpg"
name = "straight_lines2"
img_name = "test_images/" + name + ".jpg"
img = mpimg.imread(img_name)

# 1. Correct distorsion
# open distorsion matrix
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
dir_bin = dir_threshold(undist, sobel_kernel=15, thresh=(0.9, 1.2))
hls_bin = hls_select(img, thresh=(50, 255))
white_bin = white_select(img, thresh=195)
yellow_bin = yellow_select(img)
# combine filters to a final output
combined = np.zeros_like(dir_bin)
combined[((mag_bin == 1) & (dir_bin == 1) & (hls_bin == 1)) | ((white_bin == 1) | (yellow_bin == 1))] = 1

# Plot the thresholding step
# plot_thresholds(undist, mag_bin, dir_bin,
#                 hls_bin, white_bin, yellow_bin,
#                 ((mag_bin == 1) & (dir_bin == 1) & (hls_bin == 1)), combined,
#                 ((white_bin == 1) | (yellow_bin == 1)))

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
# for i in range(len(src)):
#     cv2.line(undist, (src[i][0], src[i][1]), (src[(i + 1) % 4][0], src[(i + 1) % 4][1]), (255, 0, 0), 2)
# img_warped = cv2.warpPerspective(undist, M, (X,Y), flags=cv2.INTER_LINEAR)
# plot_warping(undist, img_warped, src)

# 4. Get polinomial fit of lines
out_img, left_fit_cf, right_fit_cf,  left_fitx, right_fitx, ploty = fit_polynomial(warped)
# leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
ok, err = sanity_chk(ploty, left_fitx, right_fitx)
print(ok, err)
# Plot polynomial result
# plot_img(out_img)

# 5. Calculate curvature
curv_left, curv_right = find_curv(ploty, left_fit_cf, right_fit_cf)
road_curv = (curv_left + curv_right) / 2
lane_w = (right_fitx[-1] - left_fitx[-1]) * 3.7/700
offset = (((right_fitx[-1] + left_fitx[-1]) - img.shape[1]) / 2) * 3.7/700

print("base dist:  ", right_fitx[-1] - left_fitx[-1])
print("upper dist: ", right_fitx[0] - left_fitx[0])
print("Curvature left: ", curv_left)
print("Curvature right: ", curv_right)
print("Lane width: ", lane_w)
print("Offset: ", offset)

# 6. Plot fitted lanes into original image
# create an image to draw the lines on
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

# add text
curv_txt = "Radius of curvature: {0:.0f}m".format(road_curv)
side = {True: "left", False: "right"}
offset_txt = "Car is {0:.2f}m {1:s} of center".format(offset, side[offset>0])

cv2.putText(result, curv_txt, (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.putText(result, offset_txt, (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

plot_img(result)

mpimg.imsave("output_images/"+ name + "_output.jpg", result)