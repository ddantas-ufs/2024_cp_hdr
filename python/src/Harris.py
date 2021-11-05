import Core
from Keypoint import Keypoint
from AuxFunc import ImgNormalize

def HarrisCalc(img, msobel, mgauss, sigma_x, sigma_y, k):
  Ix = Core.cv2.Sobel(img, Core.cv2.CV_32F, 1, 0, msobel, 
                      Core.cv2.BORDER_REPLICATE)
  Iy = Core.cv2.Sobel(img, Core.cv2.CV_32F, dx=0, dy=1, ksize=msobel, 
                      borderType=Core.cv2.BORDER_REPLICATE)
  Ixx = Core.cv2.GaussianBlur(Ix * Ix, ksize=[mgauss, mgauss], sigmaX=sigma_x,
                              sigmaY=sigma_y, borderType=Core.cv2.BORDER_REPLICATE)
  Iyy = Core.cv2.GaussianBlur(Iy * Iy, ksize=[mgauss, mgauss], sigmaX=sigma_x,
                              sigmaY=sigma_y, borderType=Core.cv2.BORDER_REPLICATE)
  Ixy = Core.cv2.GaussianBlur(Ix * Iy, ksize=[mgauss, mgauss], sigmaX=sigma_x,
                              sigmaY=sigma_y, borderType=Core.cv2.BORDER_REPLICATE)
  DetH = Ixx * Iyy - Ixy * Ixy
  TraceH = Ixx + Iyy
  resp_map = DetH - k * TraceH

  return resp_map

def HarrisThreshold(resp_map, min_quality):
  max_value = resp_map.max()
  th = min_quality * max_value
  kp_list = list()
  for y in range(resp_map.shape[0]):
    for x in range(resp_map.shape[1]):
      if resp_map[y, x] >= th:
        kp_list.append(Keypoint(y, x, resp_map[y, x]))
      else:
        resp_map[y, x] = 0
  
  return kp_list

def HarrisMaxSup(resp_map, kp_list, msize):
  radius = msize // 2
  kp_aux = list()
  for kp in kp_list:
    y = kp.y
    x = kp.x
    try:
      test = kp.resp >= resp_map[y - radius:y + radius + 1, x - radius:x + radius + 1]
    except:
      test = False
    if test.all():
      kp_aux.append(kp)
  
  return kp_aux

def HarrisKp(img, msobel=Core.SOBEL_SIZE, mgauss=Core.GAUSS_SIZE, sigma_x=Core.SIGMA_X, 
             sigma_y=Core.SIGMA_X, k=Core.K, min_quality=Core.MIN_QUALITY, 
             msup_size=Core.MAXSUP_SIZE):
  img_norm = ImgNormalize(img)
  img_blur = Core.cv2.GaussianBlur(src=img_norm, ksize=[mgauss, mgauss], sigmaX=sigma_x, 
                                   sigmaY=sigma_y, borderType=Core.cv2.BORDER_REPLICATE)
  resp_map = HarrisCalc(img=img_blur, msobel=msobel, mgauss=mgauss, sigma_x=sigma_x, 
                        sigma_y=sigma_y, k=k)
  kp_list = HarrisThreshold(resp_map=resp_map, min_quality=min_quality)
  kp_list = HarrisMaxSup(resp_map=resp_map, kp_list=kp_list, msize=msup_size)

  return kp_list
