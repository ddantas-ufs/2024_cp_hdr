import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def generate_lumination_map_from_image( img ):
    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    return hsv
    
"""
    MAIN
"""

root_dir_rana_pr = "F:/artur/Documents/Python Scripts/Rana/PR/"
root_dir_rana_lr = "F:/artur/Documents/Python Scripts/Rana/LR/"
img_path = "scene-6.hdr"

img_list_rana_pr = [f for f in listdir(root_dir_rana_pr) if isfile(join(root_dir_rana_pr, f))]
img_list_rana_lr = [f for f in listdir(root_dir_rana_lr) if isfile(join(root_dir_rana_lr, f))]

print( img_list_rana_pr )
print( "--------------" )
print( img_list_rana_lr )

img = cv2.imread( img_path, -1 )
cv2.normalize( img, img, 0.0, 1.0, cv2.NORM_MINMAX, -1 )

print( img )
#exit(0)

R, G, B = cv2.split( img )
#Standard
LuminanceA = (0.2126*R) + (0.7152*G) + (0.0722*B)
#Percieved A
LuminanceB = (0.299*R + 0.587*G + 0.114*B)
#Perceived B, slower to calculate
LuminanceC = cv2.sqrt(0.299*R*R + 0.587*G*G + 0.114*B*B)

L = ( LuminanceA + LuminanceB + LuminanceC ) / 3

cv2.normalize( L, L, 0.0, 255.0, cv2.NORM_MINMAX, -1 )
cv2.normalize( LuminanceA, LuminanceA, 0.0, 255.0, cv2.NORM_MINMAX, -1 )
cv2.normalize( LuminanceB, LuminanceB, 0.0, 255.0, cv2.NORM_MINMAX, -1 )
cv2.normalize( LuminanceC, LuminanceC, 0.0, 255.0, cv2.NORM_MINMAX, -1 )

cv2.imwrite( "l.png", L )
cv2.imwrite( "la.png", LuminanceA )
cv2.imwrite( "lb.png", LuminanceB )
cv2.imwrite( "lc.png", LuminanceC )

exit(0)

hsv = generate_lumination_map_from_image( img )
h, s, v = cv2.split( hsv )

#exit(0)

cv2.imwrite( "hsv.png", hsv )
cv2.imwrite( "h.png", h )
cv2.imwrite( "s.png", s )
cv2.imwrite( "v.png", v )


"""
cv2.imwrite( "luminance_map_hsv.png", hsv )

hsv[...,2].mean()
cv2.imwrite( "luminance_map_mean.png", hsv )

hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
hsv[...,2].max()
cv2.imwrite( "luminance_map_max.png", hsv )
"""