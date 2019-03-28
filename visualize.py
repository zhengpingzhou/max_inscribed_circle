import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'text.color': 'grey'})
matplotlib.rcParams.update({'font.size': 6})

from utils import *


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