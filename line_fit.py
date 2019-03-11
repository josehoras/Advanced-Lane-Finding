import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    margin = 130        # Set the width of the windows +/- margin
    minpix = 50         # Set minimum number of pixels found to recenter window

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy, nonzerox] = [255, 255, 255]
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


def find_lane_around_fit(binary_warped, x_left, x_right):
    # HYPERPARAMETER
    margin = 100
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

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


def fit_polynomial(binary_warped, plot=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    # fit if leftx, rightx are not empty
    left_fit, right_fit, left_px, right_px, ploty = fit(leftx, lefty, rightx, righty, out_img.shape[0])

    # Visualization
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy, nonzerox] = [255, 255, 255]
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Plots the left and right polynomials on the lane lines
    if plot:
        plt.plot(left_px, ploty, color='yellow')
        plt.plot(right_px, ploty, color='yellow')

    return out_img, left_fit, right_fit, left_px, right_px, ploty


def fit(leftx, lefty, rightx, righty, y_px):
    if len(leftx) > 0 and len(rightx) > 0:
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, y_px - 1, y_px)
        left_px = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_px = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        return left_fit, right_fit, left_px, right_px, ploty
    else:
        print("Empty data!")
        return


def sanity_chk(ploty, left_px, right_px):
    sane = True
    err = ""
    width = (right_px[-1] - left_px[-1]) * 3.7/700
    if width > 4.5 or width < 3:                      # Is lane width around 3.7m (+/- 0.5)?
        sane = False
        err = "No right lane distance ({0:.1f})".format(width)
    farther = len(ploty)//2
    lane_dist = right_px[farther:] - left_px[farther:]
    sigma = np.max(lane_dist) / np.min(lane_dist)
    # print("max: {0:.3f} - min: {1:.3f}".format(np.max(lane_dist), np.min(lane_dist)))
    if sigma > 1.5:    # Are lanes more or less parallel?
        sane = False
        sg = "{0:.1f}".format(sigma)
        err = err + " - " + "No parallel lanes (" + sg + ")"
    # if left_curv < 1800 or right_curv < 1800:           # Is the curvature similar?
    #     if abs(left_curv - right_curv) > 500:
    #         sane = False
    #         err = err + " - " + "No same curvature"
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
