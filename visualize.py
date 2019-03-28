import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'text.color': 'grey'})
matplotlib.rcParams.update({'font.size': 6})


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

        
def drawBg(img, dpi=100):
    fig = plt.figure(dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if img is not None: 
        if len(img.shape) == 2: 
            img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
        plt.imshow(img[:, :, ::-1])
       
       
def showEq():
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def drawCells(cells, img=None, dpi=100):
    drawBg(img, dpi)
    for c in cells:
        plt.plot([c.x1, c.x1], [c.y1, c.y2])
        plt.plot([c.x1, c.x2], [c.y2, c.y2])
        plt.plot([c.x2, c.x2], [c.y1, c.y2])
        plt.plot([c.x1, c.x2], [c.y1, c.y1])
    showEq()


def drawBoundaryPoints(boundpts, img=None, dpi=100):
    img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
    for y, x in boundpts: cv2.circle(img, (x, y), 3, (0, 0, 255))
    drawBg(img, dpi)
    showEq()


def drawCircle(circle, img=None, cells=[], dpi=100):
    (y, x), r = circle
    img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
    cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), -1)
    drawCells(cells, img, dpi)
