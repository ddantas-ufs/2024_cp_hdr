import cv2
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

def generate_lumination_map_from_image( img ):
    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    return hsv

def get_unique_values( img ):
    return np.unique( img, return_counts=True )[0] #np.array( np.unique( img, return_counts=True ) )

def get_mask_by_values( img, list_values, ini, end ):
    print( "ini:", ini, ", end:", end )
    print( "val i:", list_values[ini], ", val e:", list_values[end-1] )
    mask = np.zeros( img.shape, img.dtype )
    arr_1 = img >= list_values[ini]
    arr_2 = img <= list_values[end-1]

    mask = np.logical_and( arr_1, arr_2 ) * 1.0

    return mask

def calculate_luminance( img ):
    alpha = 0.007
    r = alpha * max( img.shape[0], img.shape[1] )
    size = int(6*r)
    if( ( size % 2) == 0 ):
        size = size+1
    
    print( "img size:", size )
    print( "img alpha:", alpha )
    print( "img r:", r )
    blur_img = cv2.GaussianBlur( img, (size, size), r )
    
    return blur_img

def calculate_cumulative_histogram( img ):
    cv2.calcHist(img, )

"""
    MAIN
"""

root_dir_rana_pr = "F:/artur/Documents/Python Scripts/Rana/PR/"
root_dir_rana_lr = "F:/artur/Documents/Python Scripts/Rana/LR/"
img_path = "scene-6.hdr"

img_list_rana_pr = [f for f in listdir(root_dir_rana_pr) if isfile(join(root_dir_rana_pr, f))]
img_list_rana_lr = [f for f in listdir(root_dir_rana_lr) if isfile(join(root_dir_rana_lr, f))]

img = cv2.imread( img_path, -1 )
cv2.normalize( img, img, 0.0, 1.0, cv2.NORM_MINMAX, -1 )

# CALCULATE LUMINANCE
L = calculate_luminance( img )

# NORMALIZING
cv2.normalize( L, L, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3 )

print( "##### Types #####" )
print( "CV:", cv2.CV_8UC3 )
print( "AR:", L.dtype )

L_heat = cv2.applyColorMap( L, cv2.COLORMAP_JET )

cv2.imwrite( "LuminanceMap.png", L )
cv2.imwrite( "LuminanceMapHeatmap.png", L_heat )

#print( img )

#arr = get_unique_values( img )

#print( arr.shape )
exit(0)

pixel_occurrences = get_unique_values( L )
total_occurrences = len( pixel_occurrences )

pixel_occurrences.sort()

g1 = int(total_occurrences) , int(total_occurrences/2)+1
g2 = int(total_occurrences/2), int(total_occurrences/3)+1
g3 = int(total_occurrences/3), 0


print( "Mask 1" )
mask1 = get_mask_by_values( L, pixel_occurrences, g1[1], g1[0] )
print( "Mask 2" )
mask2 = get_mask_by_values( L, pixel_occurrences, g2[1], g2[0] )
print( "Mask 3" )
mask3 = get_mask_by_values( L, pixel_occurrences, g3[1], g3[0] )

print( "##################################################" )
print( "Mask 1" )
print( mask1 )
print( "Mask 2" )
print( mask2 )
print( "Mask 3" )
print( mask3 )

#print( pixel_occurrences )
#print( "Shp:", L.shape )
#print( "WxH:", L.shape[0] * L.shape[1] )
#print( "Grupo 1:", g1 )
#print( "Grupo 2:", g2 )
#print( "Grupo 3:", g3 )

#cv2.normalize( mask1, mask1, 0.0, 255.0, cv2.NORM_MINMAX, -1 )
#cv2.normalize( mask2, mask2, 0.0, 255.0, cv2.NORM_MINMAX, -1 )
#cv2.normalize( mask3, mask3, 0.0, 255.0, cv2.NORM_MINMAX, -1 )
mask1 = mask1 * 255
mask2 = mask2 * 255
mask3 = mask3 * 255

print( "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" )
print( "Mask 1" )
print( mask1 )
print( "Mask 2" )
print( mask2 )
print( "Mask 3" )
print( mask3 )


l_eq = cv2.equalizeHist( L )


cv2.imwrite( "ROIh.png", mask1 )
cv2.imwrite( "ROIm.png", mask2 )
cv2.imwrite( "ROIl.png", mask3 )

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