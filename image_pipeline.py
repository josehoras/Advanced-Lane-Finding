import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# Threshold functions
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(abs_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    mag_binary = np.zeros_like(gray)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(gray)
    binary_output[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return binary_output


def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s = hls[:, :, 2]
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    # binary_output = np.copy(img) # placeholder line
    return binary_output

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
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
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


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit_cf = np.polyfit(lefty, leftx, 2)
    right_fit_cf = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit_cf[0] * ploty ** 2 + left_fit_cf[1] * ploty + left_fit_cf[2]
        right_fitx = right_fit_cf[0] * ploty ** 2 + right_fit_cf[1] * ploty + right_fit_cf[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Visualization
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit_cf, right_fit_cf, left_fitx, right_fitx, ploty


def calculate_curvature(left_fit_cf, right_fit_cf, ploty):
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


# Plotting functions
def plot_thresholds(undist, gradx, grady, mag_binary, dir_binary, hls_binary, combined):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12, 8))
    f.tight_layout()
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
        ax.axis('off')
    ax1.set_title('Original Image', fontsize=18)
    ax1.imshow(undist)
    ax2.set_title('Grad. X', fontsize=18)
    ax2.imshow(gradx, cmap='gray')
    ax3.set_title('Grad. Y', fontsize=18)
    ax3.imshow(grady, cmap='gray')
    ax4.set_title('Magnitud', fontsize=18)
    ax4.imshow(mag_binary, cmap='gray')
    ax5.set_title('Direction', fontsize=18)
    ax5.imshow(dir_binary, cmap='gray')
    ax6.set_title('Saturation', fontsize=18)
    ax6.imshow(hls_binary, cmap='gray')
    ax8.set_title('Combined', fontsize=18)
    ax8.imshow(combined, cmap='gray')
    plt.subplots_adjust(left=0.01, right=1, top=0.9, bottom=0.)
    plt.show()
    return


def plot_warping(original, warp):
    # Uncomment this code to visualize the region of interest
    for i in range(len(src)):
        cv2.line(original, (src[i][0], src[i][1]), (src[(i+1)%4][0], src[(i+1)%4][1]), 1, 2)
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.imshow(original)
    ax2.imshow(warp)
    plt.show()
    return


def plot_img(image):
    plt.imshow(image)
    plt.show()

# *** PIPELINE ***

# Open distorsion matrix
try:
    saved_dist = pickle.load(open('calibrate_camera.p', 'rb'), encoding='latin1')
    mtx = saved_dist['mtx']
    dist = saved_dist['dist']
except (OSError, IOError):  # No progress file yet available
    print("No saved distorsion data. Run camera_calibration.py")

# Get one image
img_name = "test_images/straight_lines2.jpg"
img = mpimg.imread(img_name)

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
         [685, 450],
         [595, 450]])
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
out_img, left_fit_cf, right_fit_cf,  left_fitx, right_fitx, ploty = fit_polynomial(warped)
# Plot polynomial result
plot_img(out_img)

# 5. Calcutale curvature
curv_left, curv_right = calculate_curvature(left_fit_cf, right_fit_cf, ploty)
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
