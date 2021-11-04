import Core

def ImgNormalize(img):
  if img.dtype == 'uint8':
    return img / Core.LDR_MAX_RANGE
  else:
    return img / Core.HDR_MAX_RANGE
    