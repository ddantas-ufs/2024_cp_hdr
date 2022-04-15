import cv2
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
        return  255 - img

"""
    GENERATE ILUMINATION MAP FROM IMAGE
"""
def generate_lumination_map_from_image( img ):
    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    return hsv

"""
    CALCULATES HEAT MAP FROM IMAGE
"""
def calculate_heatmap( img ):
    heat = np.uint8( img )
    heat = cv2.applyColorMap( heat, cv2.COLORMAP_JET )

    return heat

def applyROIMask( img, ROI ):
    res = np.zeros( img.shape, img.dtype )
    ROI_mask = np.uint8( ROI/255 )

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
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return np.asarray(gray, dtype=np.uint8)
    return None

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

"""
    CALCULATE ILLUMINATION SUBREGIONS OF IMAGE
    @param img_path: directory where image is stored
    @param img_name: image name
    @param img_mask_path: absolute path to image ROI (if none, all 255 image is used)
    @param img_out_dir: path to directory where output must be stored
    @oaram regions: amount of ilumination images
"""
def calculate_subregions( img_path, img_name, img_mask_path, img_out_dir, regions ):
    print( "Method: Calculate Subregions" )

    img_name_str = Path(img_name).stem

    interval_size = int( 255/regions )

    img = imread( img_path+img_name )
    imgMask = None

    if( img_mask_path is not None ):
        imgMask = rgb2gray( imread(img_mask_path) )
    else:
        imgMask = np.ones( (img.shape[0], img.shape[1]), np.uint8 )
    
    cv2.normalize( img, img, 0.0, 1.0, cv2.NORM_MINMAX, -1 )

    # CALCULATE LUMINANCE
    L = calculate_luminance( img )

    # NORMALIZING
    cv2.normalize( L, L, 0, 255, cv2.NORM_MINMAX, -1 )
    if( imgMask is not None ):
        cv2.normalize( imgMask, imgMask, 0, 255, cv2.NORM_MINMAX, -1 )

    # CREATE IMAGE HEATMAP
    heat = calculate_heatmap( L )

    L = applyROIMask( L, imgMask )
    L_gray = applyROIMask( rgb2gray( L ), imgMask )

    # CALCULATE HISTOGRAM EQUALIZATION
    L_histeq = histeqRGB( L )
    L_histeq_gray = histeq( L_gray )

    thresholds = []
    for i in range( 0, regions ):
        ini = i*interval_size
        end = i*interval_size+interval_size
        thresholds.append( thresh( L_histeq_gray, ini, end ) )

    ROI_list = []
    if( imgMask is not None ):
        for i in range( 0, regions ):
            ROI_list.append( applyROIMask(thresholds[i], imgMask ) * 255 ) #np.logical_and( imgMask, thresholds[i] ) * 255 )
            #ROI_list.append( np.logical_and( imgMask, thresholds[i] ) * 255 )
    else:
        ROI_list = ROI_list + thresholds

    # CREATING UNITED ROI
    ROI_all = np.zeros( (img.shape[0], img.shape[1]), img.dtype )
    for i in range( 0, regions ):
        ROI_all = np.logical_or( ROI_all, ROI_list[i] ) * 255

    # OBTAINING ROIs
    ROI_segmented_regions = np.zeros( (img.shape[0],img.shape[1],3), np.uint8 )
    segment_levels = int(128 / regions)
    for i in range( 0, regions ):
        value = 128 + int( i * segment_levels )
        ROI_segmented_regions[:,:,1] = ROI_segmented_regions[:,:,1] + (ROI_list[i]/255) * value
        print( "Value now: ", segment_levels+value )

    negative_mask = negative( imgMask )
    ROI_segmented_regions[:,:,0] = ROI_segmented_regions[:,:,0] + ( negative_mask/255 ) * 128

    # SAVING ROI OF ILUMINATION REGIONS
    for i in range( 0, regions ):
        out_img_name = "_ROI_" +str(i) +".png"
        cv2.imwrite( img_out_dir + img_name_str + out_img_name, ROI_list[i] )

    # SAVING OTHER IMAGES    
    #cv2.imwrite( img_out_dir + img_name_str + "_ROI_completo.png", ROI_all )
    cv2.imwrite( img_out_dir + img_name_str + "_ROI_segments.png", ROI_segmented_regions )
    cv2.imwrite( img_out_dir + img_name_str + "_LuminanceMap.png", L )
    cv2.imwrite( img_out_dir + img_name_str + "_LuminanceMapHeatmap.png", heat )
    cv2.imwrite( img_out_dir + img_name_str + "_LuminanceMapHistogramEq.png", L_histeq )

    """
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

"""
    MAIN
"""

# DIRECTORY WHERE 
root_dir_output  = "F:/artur/Documents/Python Scripts/"

# DIRECTORY WITH HDR IMAGES
root_dir_rana_pr = "F:/artur/Documents/Python Scripts/Rana/PR/"
root_dir_rana_lr = "F:/artur/Documents/Python Scripts/Rana/LR/"

# ABSOLUTE PATH OF ROI
absolute_rana_pr_mask = root_dir_rana_pr + "ROIa.png"
absolute_rana_lr_mask = root_dir_rana_lr + "ROIa.png"

img_list_rana_pr = [f for f in listdir(root_dir_rana_pr) if isfile(join(root_dir_rana_pr, f))]
img_list_rana_lr = [f for f in listdir(root_dir_rana_lr) if isfile(join(root_dir_rana_lr, f))]

#calculate_subregions( root_dir_rana_pr, "scene-0.hdr", absolute_rana_pr_mask, root_dir_rana_pr, 3 )

#"""
for img_path in img_list_rana_pr:
    if( not img_path.endswith(".png") ):
       calculate_subregions( root_dir_rana_pr, img_path, absolute_rana_pr_mask, root_dir_rana_pr, 3 )

for img_path in img_list_rana_lr:
    if( not img_path.endswith(".png") ):
        calculate_subregions( root_dir_rana_lr, img_path, None, root_dir_rana_lr, 3 )
#"""