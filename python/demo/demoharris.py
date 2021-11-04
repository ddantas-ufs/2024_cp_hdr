# -*- coding: utf-8 -*-

import os
import sys
from matplotlib import pyplot as plt

CPHDR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(CPHDR_PATH, 'python', 'src'))
IMGS_PATH = os.path.join(CPHDR_PATH, 'img')

import Core
import Harris
from Keypoint import PlotKp

img = Core.cv2.imread(IMGS_PATH + '/lena.png', Core.cv2.IMREAD_GRAYSCALE)
kp_list = Harris.HarrisKp(img)
img_kp = PlotKp(img, kp_list)
plt.imshow(Core.cv2.cvtColor(img_kp, Core.cv2.COLOR_GRAY2RGB))
plt.show()