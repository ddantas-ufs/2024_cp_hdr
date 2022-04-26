#!/usr/bin/python3

import sys
import cv2
import time
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from pathlib import Path

"""
    READS IMAGE POINTED BY filename. HDR is readed as float64 and LDR is readed as uint8.
"""
def imread( filename ):
    img = cv2.imread( filename, -1 )
    return img

def imsave( img, absolute_path ):
    print( "Saving image: {}".format(absolute_path) )
    if( np.max(img) <= 1 ):
        print( "Image in scale [0.0, 1.0]. Max: {}".format(np.max(img)) )
        cv2.normalize( img, img, 0.0, 255.0, cv2.NORM_MINMAX, -1 )

    cv2.imwrite( absolute_path, img )

"""
    RETURNS TRUE IF IS GRAYSCALE AND FALSE IF IS RGB
"""
def is_gray_image( img ):
    if( isinstance(img, np.ndarray) ):
        if( img.ndim == 2 ):
            return True
        else:
            return False
    else:
        return False

def negative( img ):
    print("method: negative")
    if( isinstance( img, np.ndarray ) ):
        if( np.max( img ) <= 1 ):
            return  1 - img
        else:
            return 255 - img

"""
    CALCULATES HEAT MAP FROM IMAGE
"""
def calculate_heatmap( img ):
    heat = np.zeros( img.shape, np.uint8 )
    
    if( np.max(img) <= 1 ):
        cv2.normalize( img, heat, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U )
    
    heat = cv2.applyColorMap( heat, cv2.COLORMAP_JET )

    return heat

def applyROIMask( img, ROI ):
    res = np.zeros( img.shape, img.dtype )
    ROI_mask = np.zeros( (ROI.shape[0],ROI.shape[1]), np.float32 )

    if( np.max(ROI) <= 1 ):
        print( "Max is <= 1: {}".format(np.max(ROI)) )
        ROI_mask = ROI
    else:
        print("Max not 1: {}".format(np.max(ROI)))
        cv2.normalize( ROI, ROI_mask, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F )

    if( is_gray_image( img ) ):
        res = img * ROI_mask
    else:
        res[:,:,0] = img[:,:,0] * ROI_mask
        res[:,:,1] = img[:,:,1] * ROI_mask
        res[:,:,2] = img[:,:,2] * ROI_mask
    return res

"""
    TRANSFORMS RGB IMAGE INTO GRAYSCALE
"""
def rgb2gray( img ):
    print("method: rgb2gray")
    if( isinstance( img, np.ndarray ) ):
        if( is_gray_image( img ) ):
            return img
        else:
            gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
            return gray
            #r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            #gray = 0.299 * r + 0.587 * g + 0.114 * b
            #return np.asarray(gray, dtype=img.dtype)
    return None

def values_inside_roi( img, ROI ):
    print("Method: values_inside_roi")
    res = []

    for i in range(0, ROI.shape[0]):
        for j in range(0, ROI.shape[1]):
            if( ROI[i,j] != 0 ):
                res.append( img[i,j] )

    print( "Values inside ROI: {}".format(len(res)) )
    return np.float32(res)
"""
    CALCULATES LUMINANCE MAP FROM IMAGE, USING RANA AND CHIU ET. AL METHOD
"""
def calculate_luminance( img ):
    alpha = 0.007
    r = alpha * max( img.shape[0], img.shape[1] )
    size = int(6*r)
    if( ( size % 2) == 0 ):
        size = size+1
    
    blur_img = cv2.GaussianBlur( img, (size, size), r )    
    return blur_img

"""
    CALCULATE HISTOGRAM FROM IMAGE
"""
def hist( img ):
    print( "method hist" )
    if( isinstance(img, np.ndarray) ):
        if( is_gray_image( img ) ):
            return np.reshape( np.bincount( img.ravel(), minlength=256 ), (256, 1) )
        else:
            hist = np.zeros( (256, 3), np.uint32 )
            hist[:, 0] = np.reshape( np.bincount( img[:, :, 0].ravel(), minlength=256 ), (256) ).astype( np.uint32 )
            hist[:, 1] = np.reshape( np.bincount( img[:, :, 1].ravel(), minlength=256 ), (256) ).astype( np.uint32 )
            hist[:, 2] = np.reshape( np.bincount( img[:, :, 2].ravel(), minlength=256 ), (256) ).astype( np.uint32 )
            
            return hist
    return None

"""
    CALCULATE IMAGE HISTOGRAM EQUALIZATION
"""
def histeq( img ):
    print( "method histeq" )
    if( isinstance(img, np.ndarray) ):
        if( not is_gray_image( img ) ):
            img = rgb2gray( img )
        
        # CALCULATE CDF VALUES
        cdf = np.cumsum( hist( img ).flatten() )
        
        # MASKING 0-VALUES
        cdf_m = np.ma.masked_equal( cdf, 0 )

        # CALCULATING CUMULATIVE HISTOGRAM
        cdf_m = ( ( cdf_m - cdf_m.min() ) * 255 ) / ( cdf_m.max() - cdf_m.min() )
        
        # NORMALIZING [0,255]
        cdf_f = np.ma.filled( cdf_m, 0 ).astype( 'uint8' )
        
        # TRANSFORMING INTO IMAGE
        img_ret = cdf_f[img]
        
        return img_ret

"""
    CALCULATES HISTOGRAM EQUALIZATION OF RGB IMAGE
"""
def histeqRGB( img ):
    img_eq = np.zeros( img.shape, np.uint8 )
    img_eq[:,:,0] = histeq( np.uint8( img[:,:,0] ) )
    img_eq[:,:,1] = histeq( np.uint8( img[:,:,1] ) )
    img_eq[:,:,2] = histeq( np.uint8( img[:,:,2] ) )
    return img_eq

"""
    CALCULATES TRESHOLD OF INTERVAL
"""
def thresh( img, tr_min, tr_max ):
    #print("method: threshold")
    #print("thresh min:", tr_min, "thresh max:", tr_max)
    if( isinstance( img, np.ndarray ) ):
        if( tr_min == tr_max ):
            img_tresh = (img == tr_min) * 1
            return img_tresh
        elif( is_gray_image( img ) ):
            img_thresh = np.logical_and( img > tr_min, img <= tr_max ) * 1
            return img_thresh
        else:
            img_bool_r = np.logical_and( img[:, :, 0] >= tr_min, img[:, :, 0] < tr_max )
            img_bool_g = np.logical_and( img[:, :, 1] >= tr_min, img[:, :, 1] < tr_max )
            img_bool_b = np.logical_and( img[:, :, 2] >= tr_min, img[:, :, 2] < tr_max )
            img_thresh = np.zeros( (img.shape[0], img.shape[1], img.shape[2]), np.uint8 )
            
            img_thresh[:, :, 0] = img_bool_r * 1
            img_thresh[:, :, 1] = img_bool_g * 1
            img_thresh[:, :, 2] = img_bool_b * 1
            
            return img_thresh
    return None

def calculate_subregions_by_pixels_without_histeq( path_to_img, img_mask_path, img_out_dir, regions ):
    print( "Method: Calculate Subregions" )

    print( "Image Name:", path_to_img )
    print( "ROI Path:", img_mask_path )

    img_name_str = Path(path_to_img).stem

    img = imread( path_to_img )
    #img = imread( img_path+img_name )
    ROI0_1 = np.ones( (img.shape[0], img.shape[1]), np.float32 )
    ROI0_255 = np.ones( (img.shape[0], img.shape[1]), np.uint8 ) * 255

    if( img_mask_path is not None ):
        print( "-> ROI being used" )
        ROI0_255 = rgb2gray( imread(img_mask_path) )
    
    print( "--------> {}".format(cv2.countNonZero( ROI0_1 )) )
    # NORMALIZING IMAGE AND ROI
    #cv2.normalize( ROI0_255, ROI0_1, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32FC1 )
    ROI0_1 = ROI0_255 / 255
    cv2.normalize( img, img, 0.0, 1.0, cv2.NORM_MINMAX, -1 )

    print( "--------> {}".format(cv2.countNonZero( ROI0_1 )) )

    print( "Shape Image: {}".format( img.shape ) )
    print( "Shape ROI  : {}".format( ROI0_1.shape ) )
    #return 0

    L = calculate_luminance( img )
    cv2.normalize( L, L, 0.0, 1.0, cv2.NORM_MINMAX, -1 )

    # CREATE IMAGE HEATMAP
    heat = calculate_heatmap( L )

    L_gray = rgb2gray( L )
    L_gray_ROI = L_gray

    if( img_mask_path is not None ):
        L = applyROIMask( L, ROI0_1 )
        L_gray_ROI = values_inside_roi(L_gray, ROI0_1)

    unique_occurrences, maskedHistogram = np.unique( L_gray_ROI, return_counts=True )

    amount_of_bins = len(unique_occurrences)

    print( "Unique Occurrences: {}".format(unique_occurrences) )
    print( "Amount of bins: {}".format(amount_of_bins) )
    print( "Image: Min: {}, Max: {}".format( np.min(img), np.max(img) ) )
    print( "Luminance: Min: {}, Max: {}".format( np.min(L_gray), np.max(L_gray) ) )
    print( "ROI Luminance: Min: {}, Max: {}".format( np.min(L_gray_ROI), np.max(L_gray_ROI) ) )

    total_pixels = cv2.countNonZero( ROI0_1 )

    print( maskedHistogram )
    print( maskedHistogram.shape )
    print( "Multiply - X*Y :", L_gray.shape[0]*L_gray.shape[1] )
    print( "Sum OCV Histogram:", np.sum( maskedHistogram ) )
    print( "Valid Pixels:", total_pixels )

    pixels_in_interval = int(total_pixels / regions)
    print( "Pixels in interval: {}".format(pixels_in_interval) )
    print( maskedHistogram )
    
    limits = []
    pixel_sum = 0
    for i in range( 0, amount_of_bins ):
        pixel_sum = pixel_sum + maskedHistogram[i]
        if( pixel_sum >= pixels_in_interval or i == amount_of_bins-1 ):
            print( "Parcial count at index {}: {}".format(i, pixel_sum) )
            pixel_sum = 0
            limits.append( i )

    print( "Limits: {}".format(limits) )
    
    print( "Limits: {}".format(limits) )
    print( "Total of Pixels: {}".format(total_pixels) )
    print( "Pixels in Each interval: {}".format(pixels_in_interval) )

    thresholds = []
    for i in range( 0, len(limits) ):
        ini = .0
        end = .0

        if( i == 0 ):
            end = unique_occurrences[ limits[i] ] 
        elif( i == len(limits)-1 ):
            ini = unique_occurrences[ limits[i-1] ]
            end = unique_occurrences[ limits[i] ] + 1
        else:
            ini = unique_occurrences[ limits[i-1] ] #(limits[i-1]+1)/amount_of_bins
            end = unique_occurrences[ limits[i] ] #(limits[i]+1)/amount_of_bins

        print("ini:", ini, "fim:", end)
        thresholds.append( thresh( L_gray, ini, end ) )

    #exit(0)
    ROI_list = []
    if( img_mask_path is not None ):
        for i in range( 0, regions ):
            ROI_list.append( applyROIMask( thresholds[i], ROI0_1 ) ) 
    else:
        ROI_list = ROI_list + thresholds

    # OBTAINING ROIs
    ROI_segmented_regions = np.zeros( (img.shape[0],img.shape[1],3), np.float32 )
    segment_levels = 0.58/regions
    actual_level = 0.58
    for i in range(0, len(ROI_list)):
        print( "Value now at ROI {}: {}".format(i, actual_level) )
        roi = ROI_list[i]
        ROI_segmented_regions[:,:,1] += ( roi * actual_level )
        actual_level += segment_levels
    
    # SEGMENTATION
    if( img_mask_path is not None ):
        negative_mask = negative( ROI0_1 )
        ROI_segmented_regions[:,:,0] += ( negative_mask * 0.44 )

    # SAVING ROI OF ILUMINATION REGIONS
    for i in range( 0, regions ):
        out_img_name = "_ROI_" +str(i) +".png"
        imsave( ROI_list[i], img_out_dir + img_name_str + out_img_name )

    # SAVING OTHER IMAGES
    imsave( ROI_segmented_regions, img_out_dir + img_name_str + "_ROI_segments.png" )
    imsave( heat, img_out_dir + img_name_str + "_LuminanceMapHeatmap.png" )
    imsave( L, img_out_dir + img_name_str + "_LuminanceMap.png" )

"""
    MAIN

    How to call this script:
        path_to_img path_to_ROI output_path subregions
"""
# ARGS
path_to_img = None
path_to_ROI = None
output_path = None
subregions  = None

if( len(sys.argv) == 4 ):
    path_to_img = sys.argv[1]
    path_to_ROI = None
    output_path = sys.argv[2]
    subregions  = int(sys.argv[3])
else:
    path_to_img = sys.argv[1]
    path_to_ROI = sys.argv[2]
    output_path = sys.argv[3]
    subregions  = int(sys.argv[4])

calculate_subregions_by_pixels_without_histeq( path_to_img, path_to_ROI, output_path, subregions )
#exit(0)
print( "Quantidade de argumentos: {}".format( len(sys.argv) ) )
#for i in range( 0, len(sys.argv) ):
#    print( "Argumento {}: {}".format( i, sys.argv[i] ) )

print( "path_to_img: {}".format( path_to_img ) )
print( "path_to_ROI: {}".format( path_to_ROI ) )
print( "output_path: {}".format( output_path ) )
print( "subregions : {}".format( subregions  ) )
print( " ----------------------------------- " )
