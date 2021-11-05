import Core

def ImgNormalize(img):
  if img.dtype == 'uint8':
    return Core.np.float32(img / Core.LDR_MAX_RANGE)
  else:
    return Core.np.float32(img / Core.HDR_MAX_RANGE)
    