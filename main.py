import cv2
import numpy as np

from max_inscribed_circle import *
from visualize import *

mask = cv2.imread('./mask.png', 0)
circle, cells = maxInscribedCircle(mask, return_cells=True)
drawCircle(circle, mask, cells, dpi=100)
