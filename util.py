import matplotlib

def video_rgb_to_hsv(video):
  T=video.shape[0]
  for t in range(T):
    video[t,:,:,:]=matplotlib.colors.rgb_to_hsv(video[t,:]/255)


def video_hsv_to_rgb(video):
  T=video.shape[0]
  for t in range(T):
    video[t,:,:,:]=matplotlib.colors.hsv_to_rgb(video[t,:])*255
