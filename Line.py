import numpy as np

# Class Line
class Line():
    def __init__(self):
        # number of frames to keep in history
        self.nframes = 4
        # history of x values lane and its average
        self.fit_x = None
        self.fit_x_hist = []
        # y values for detected line pixels
        self.fit_y = None
        # history of polynomial coefficients and its average
        self.fit_hist = []
        self.fit_avg = None
        # history of radius of curvature of the line in meters and its average
        self.curv_avg = None
        self.curv_hist = []
        # lane position closest to the vehicle in x
        self.base_pos = None

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

        self.base_pos = self.fit_x[-1]