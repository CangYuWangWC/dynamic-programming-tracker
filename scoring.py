import numpy as np
from scipy.stats import multivariate_normal
from scipy import ndimage

def skin_pixel_scorer(video,mu,cov):
  T,h,w,c=video.shape
  score=np.zeros((T,h,w))

  for t in range(T):
    score[t,:,:]= multivariate_normal.pdf(video[t,:,:,:],mu,cov)

  mean=score.mean()
  std=score.std()
  threshold = mean + 0.2*std
  print("Pixel score treshold: %f" % threshold)
  score[score<threshold]=0
  score=ndimage.grey_erosion(score,(1, 10, 10))
  #score= ndimage.gaussian_filter(score, sigma=(0,3,3), order=0)
  return score

def euclidean_movement_score(video):
  T,h,w,c=video.shape
  score=np.zeros((T,h,w))

  delta=np.diff(video,n=1,axis=0)
  delta=np.square(delta)
  delta=np.sqrt(np.sum(delta,axis=3))
  print(delta.shape)
  score[1:,:,:]=delta

#   for t in range(T-1):
#     delta=np.square(video[t+1,:,:,:]-video[t,:,:,:])
#     score[t,:,:]= np.sqrt(np.sum(delta,axis=2))
    # if (t % 5==0):
    #   plt.clf()
    #   plt.imshow(score[t,:,:])
    #   plt.colorbar()
    #   plt.pause(0.1*1.0/60.0)
  mean=score.mean()
  std=score.std()
  threshold = mean - 0.1*  std
  print("Movement score treshold: %f" % threshold)
  score[score<threshold]=0
  score=ndimage.grey_erosion(score,(1, 20, 20))
  score= ndimage.gaussian_filter(score, sigma=(0,3,3), order=0)
  return score

def manhattan_movement_score(video):
  T,h,w,c=video.shape
  score=np.zeros((T,h,w))

  delta=np.diff(video,n=1,axis=0)
  delta=np.abs(delta)
  delta=np.sum(delta,axis=3)
  print(delta.shape)
  score[1:,:,:]=delta

  mean=score.mean()
  std=score.std()
  threshold = mean - 0.1*  std
  print("Movement score treshold: %f" % threshold)
  score[score<threshold]=0
  score=ndimage.grey_erosion(score,(1, 20, 20))
  score= ndimage.gaussian_filter(score, sigma=(0,3,3), order=0)
  return score

def calculate_local_score(video,scorers,weights):

    score=scorers[0](video)*weights[0]
    local_score=score
    scores=[score]
    for i in range(1,len(scorers)):
        score=scorers[i](video)*weights[i]
        local_score=local_score+score
        scores.append(score)
    return local_score,scores
