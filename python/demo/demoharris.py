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

img = Core.cv2.imread(filename=IMGS_PATH + '/lena.png', flags=Core.cv2.IMREAD_GRAYSCALE)
kp_list = Harris.HarrisKp(img=img)
img_kp = PlotKp(img=img, kp=kp_list)
plt.imshow(Core.cv2.cvtColor(src=img_kp, code=Core.cv2.COLOR_GRAY2RGB))
plt.show()
