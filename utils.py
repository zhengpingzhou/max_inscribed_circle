import cv2
import matplotlib
import numpy as np 
from skimage.color import hsv2rgb, gray2rgb, rgb2gray

import matplotlib.pyplot as plt
matplotlib.rcParams.update({'text.color': 'grey'})

# ------------------------------------------------------------------------------
# Display 1 or more images
# ------------------------------------------------------------------------------
def imshow(data, titles=[], dpi=80):
    if type(data) == list:
        for i in range(len(data)):
            if len(data[i].shape) == 2:
                data[i] = data[i][:, :, np.newaxis]
                data[i] = np.tile(data[i], [1, 1, 3])
            if data[i].dtype in {float, np.float32, np.float64}:
                data[i] = np.uint8(data[i] * 255)
        assert len(titles) == 0 or len(titles) == len(data)
        fig, axs = plt.subplots(nrows=1, ncols=len(data), dpi=dpi)
        for ax in axs: ax.set_axis_off()
        for i, im in enumerate(data):
            axs[i].imshow(im[:, :, ::-1])
            if len(titles) > 0:
                axs[i].set_title(titles[i])
        plt.show()
    else:
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = np.tile(data, [1, 1, 3])
        if data.dtype in {float, np.float32, np.float64}:
            data = np.uint8(data * 255)
        fig = plt.figure(dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(data[:, :, ::-1])
        plt.show()

# ------------------------------------------------------------------------------
# visualization
# ------------------------------------------------------------------------------
def color_overlay(img, maps, h_shift, indices, one_hot, n_colors=None, start_color=0, intensities=None):
    n_colors = len(indices) if n_colors is None else n_colors

    color_palette = hsv2rgb(np.dstack([
        np.arange(h_shift, 1, 1 / n_colors),
        0.7 * np.ones(n_colors),
        0.3 * np.ones(n_colors)
    ]))[0]

    out = gray2rgb(rgb2gray(img))  # desaturate
    out *= 0.5

    names = []
    masks = []

    for i, (name, idx) in enumerate(indices):
        m = maps == idx if one_hot else maps[:, :, idx]
        # m = np.clip(m, 0, 1)
        col = color_palette[start_color + i]

        if intensities is not None:
            col = col * intensities[i]

        out = np.clip(out + col * m[:, :, None], 0, 1)
        names.append(name)
        masks.append(col * m[:, :, None])

    imshow(masks, names, dpi=200)

    out = (255 * out).astype('uint8')
    return out

# ------------------------------------------------------------------------------
# logging to stdout&file
# ------------------------------------------------------------------------------
class Logger(object):
    def __init__(self, outfile=None):
        self.outfile = outfile
        self.fout = None

    def LOG(self, info):
        if self.fout is None:
            self.fout = open(self.outfile, 'w')
        self.fout.write(str(info) + '\n')
        self.fout.flush()
        print(info)
        
# ------------------------------------------------------------------------------
# max inscribed circle
# ------------------------------------------------------------------------------
def max_inscribed_circle(mask):
    """
    Args:
        + mask: binary (0 or 1)
    Returns:
        + (c, r)]
    1. find countour
    2. for each countour: find circle
    3. return the maximum circle
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
