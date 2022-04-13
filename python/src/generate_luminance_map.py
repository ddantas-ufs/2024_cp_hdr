import cv2
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from pathlib import Path


def imread( filename ):
    img = cv2.imread( filename, -1 )
    return img

def is_gray_image( img ):
    if( isinstance(img, np.ndarray) ):
        if( img.ndim == 2 ):
            return True
        else:
            return False
    else:
        return False

def generate_lumination_map_from_image( img ):
    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    return hsv

def get_unique_values( img ):
    return np.unique( img, return_counts=True )[0] #np.array( np.unique( img, return_counts=True ) )

def calculate_cumulative_histogram( img ):
    return cv2.calcHist( img )

def calculate_heatmap( img ):
    heat = np.uint8( img )
    heat = cv2.applyColorMap( heat, cv2.COLORMAP_JET )

    return heat

def rgb2gray( img ):
    print("method: rgb2gray")
    if( isinstance( img, np.ndarray ) ):
        if( is_gray_image( img ) ):
            return img
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return np.asarray(gray, dtype=np.uint8)
    return None

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

def histeq( img ):
    print( "method histeq" )
    if( isinstance(img, np.ndarray) ):
        if( not is_gray_image( img ) ):
            img = rgb2gray( img )
        
        # CALCULANDO CDF
        cdf = np.cumsum( hist( img ).flatten() )
        
        # MASCARANDO PIXELS 0
        cdf_m = np.ma.masked_equal( cdf, 0 )

        # CALCULA HISTOGRAMA ACUMULADO [T(rk)]
        cdf_m = ( ( cdf_m - cdf_m.min() ) * 255 ) / ( cdf_m.max() - cdf_m.min() )
        
        # NORMALIZA HISTOGRAMA [0, 255]
        cdf_f = np.ma.filled( cdf_m, 0 ).astype( 'uint8' )
        
        # FAZ A TRANSFORMACAO DA IMAGEM
        img_ret = cdf_f[img]
        
        return img_ret

def histeqRGB( img ):
    img_eq = np.zeros( img.shape, np.uint8 )
    img_eq[:,:,0] = histeq( np.uint8( img[:,:,0] ) )
    img_eq[:,:,1] = histeq( np.uint8( img[:,:,1] ) )
    img_eq[:,:,2] = histeq( np.uint8( img[:,:,2] ) )
    return img_eq

def thresh( img, tr_min, tr_max ):
    print("method: threshold")
    print("thresh min:", tr_min, "thresh max:", tr_max)
    if( isinstance( img, np.ndarray ) ):
        if( is_gray_image( img ) ):
            #img_thresh = (img >= tr) * 255
            img_thresh = np.logical_and( img >= tr_min, img <= tr_max ) * 255
            return img_thresh
        else:
            img_bool_r = np.logical_and( img[:, :, 0] >= tr_min, img[:, :, 0] <= tr_max )
            img_bool_g = np.logical_and( img[:, :, 1] >= tr_min, img[:, :, 1] <= tr_max )
            img_bool_b = np.logical_and( img[:, :, 2] >= tr_min, img[:, :, 2] <= tr_max )
            img_thresh = np.zeros( (img.shape[0], img.shape[1], img.shape[2]), np.uint8 )
            
            img_thresh[:, :, 0] = img_bool_r * 255
            img_thresh[:, :, 1] = img_bool_g * 255
            img_thresh[:, :, 2] = img_bool_b * 255
            
            return img_thresh
    return None

def calculate_subregions( img_path, img_name, img_mask_path, img_out_dir ):
    print( "Method: Calculate Subregions" )

    print( "img_path :", img_path )
    print( "img_name :", img_name )
    print( "img_mask_path :", img_mask_path )
    print( "img_out_dir :", img_out_dir )

    img_name_str = Path(img_name).stem

    img = imread( img_path+img_name )
    imgMask = np.ones( (img.shape[0], img.shape[1]), img.dtype )

    if( img_mask_path is not None ):
        imgMask = rgb2gray( imread(img_mask_path) )
    else:
        imgMask = None 
    
    cv2.normalize( img, img, 0.0, 1.0, cv2.NORM_MINMAX, -1 )

    # CALCULATE LUMINANCE
    L = calculate_luminance( img )

    # NORMALIZING
    cv2.normalize( L, L, 0, 255, cv2.NORM_MINMAX, -1 )
    if( imgMask is not None ):
        cv2.normalize( imgMask, imgMask, 0, 255, cv2.NORM_MINMAX, -1 )

    # CREATE IMAGE HEATMAP
    heat = calculate_heatmap( L )

    L_histeq = histeqRGB( L )
    L_histeq_gray = histeq( rgb2gray(L) )

    thrl = thresh( L_histeq_gray, 0, 84 )
    thrm = thresh( L_histeq_gray, 85, 169 )
    thrh = thresh( L_histeq_gray, 170, 255 )

    if( imgMask is not None ):
        ROIl = np.logical_and( imgMask, thrl ) * 255
        ROIm = np.logical_and( imgMask, thrm ) * 255
        ROIh = np.logical_and( imgMask, thrh ) * 255
    else:
        ROIl = thrl
        ROIm = thrm
        ROIh = thrh

    ROI = np.logical_or( ROIl, ROIm ) * 255
    ROI = np.logical_or( ROI, ROIh ) * 255

    ROISeg = np.zeros( (ROI.shape[0], ROI.shape[1], 3), np.uint8 )
    ROISeg[:,:,0] = ROIl
    ROISeg[:,:,1] = ROIm
    ROISeg[:,:,2] = ROIh

    cv2.imwrite( img_out_dir + img_name_str + "_ROIh.png", ROIh )
    cv2.imwrite( img_out_dir + img_name_str + "_ROIm.png", ROIm )
    cv2.imwrite( img_out_dir + img_name_str + "_ROIl.png", ROIl )
    cv2.imwrite( img_out_dir + img_name_str + "_ROI.png", ROI )
    cv2.imwrite( img_out_dir + img_name_str + "_ROISegments.png", ROISeg )
    cv2.imwrite( img_out_dir + img_name_str + "_LuminanceMap.png", L )
    cv2.imwrite( img_out_dir + img_name_str + "_LuminanceMapHeatmap.png", heat )
    cv2.imwrite( img_out_dir + img_name_str + "_LuminanceMapHistogramEq.png", L_histeq )
    cv2.imwrite( img_out_dir + img_name_str + "_LuminanceMapHistogramEqGray.png", L_histeq_gray )

"""
    MAIN
"""

root_dir_output  = "F:/artur/Documents/Python Scripts/"
root_dir_rana_pr = "F:/artur/Documents/Python Scripts/Rana/PR/"
root_dir_rana_lr = "F:/artur/Documents/Python Scripts/Rana/LR/"
#img_path = "scene-6.hdr"

img_list_rana_pr = [f for f in listdir(root_dir_rana_pr) if isfile(join(root_dir_rana_pr, f))]
img_list_rana_lr = [f for f in listdir(root_dir_rana_lr) if isfile(join(root_dir_rana_lr, f))]

#for img_path in img_list_rana_pr:
#    if( not "ROIa.png" == img_path ):
#       calculate_subregions( root_dir_rana_pr, img_path, root_dir_rana_pr+"ROIa.png", root_dir_rana_pr )

for img_path in img_list_rana_lr:
    if( not "ROIa.png" == img_path ):
        calculate_subregions( root_dir_rana_lr, img_path, None, root_dir_rana_lr )

#calculate_subregions( root_dir_output, img_path, root_dir_rana_pr+"ROIa.png", "F:/artur/Documents/Python Scripts/Rana/" )