import os
import numpy as np
import skimage.io
import skimage
from scipy import ndimage
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim

from scipy.stats import multivariate_normal

import skvideo.io
import skvideo.datasets



def jump_penalty_euclidean(p1,p2):
  d=p1-p2
  return np.sum(d**2)

class DPTracker:
  def __init__(self,neighbourhood_size):
    self.neighbourhood_size=neighbourhood_size


  def track(self,video,movement_scorer,movement_vs_pixel_weight,pixel_scorer,jump_penalty,jump_penalty_weight):
    video = ndimage.gaussian_filter(video, sigma=(1, 2, 2, 0), order=0)
    T,h,w,c=video.shape
    self.backtracking_path=np.zeros((T-1,h,w,2),dtype=int) # 2 => (x,y) for T-1
    self.score=np.ones((T,h,w))*(-1) # global score
    self.movement_score=movement_scorer(video)
    self.pixel_score=pixel_scorer(video)
    n=self.neighbourhood_size
    n2=n+1
    jump_penalty_matrix=jump_penalty_weight*self.generate_jump_penalty_matrix(jump_penalty,n)
    self.score[0,:,:]=self.pixel_score[0,:,:]

    local_score = movement_vs_pixel_weight * self.movement_score + \
                 (1 - movement_vs_pixel_weight) * self.pixel_score[1:-1, :, :]
    for t in range(1,T):
      print("Calculating score for frame %d / %d " % (t,T-1))
      for i in range(n+1, h-n-1):
        for j in range(n+1, w-n-1):
          neighbourhood_score=local_score[i-n:i+n2,j-n:j+n2]
          previous_score=neighbourhood_score-jump_penalty_matrix

          index=np.argmax(previous_score)
          best_score=previous_score.flat[index]
          relative_x,relative_y=np.unravel_index(index, jump_penalty_matrix.shape)
          x= relative_x-neighbourhood_size+i
          y = relative_y - neighbourhood_size + j
          self.backtracking_path[t-1, i, j, :] =np.array([x,y])
          self.score[t,i,j]=self.pixel_score[t,i,j]+np.mean(previous_score) #best_score

      # if t % 3 ==0:
      #   #TODO show 4 panels: movement score, pixel score, previous score, score
      #   plt.clf()
      #   plt.imshow(self.score[t,:,:])
      #   plt.colorbar()
      #   plt.pause(1.0/60.0)
    self.path=np.zeros((T,2),dtype=int) # 2 => (x,y)
    best_index_last_frame = np.argmax(self.score[-1,:,:])
    x,y= np.unravel_index(best_index_last_frame , self.score[-1,:,:].shape)
    self.path[-1,:]=np.array([x,y])
    for t in reversed(range(T-1)):
      x,y= tuple(self.path[t+1,:])
      self.path[t, :] = self.backtracking_path[t,x,y,:]
    return self.path

  def generate_jump_penalty_matrix(self, jump_penalty, neighbourhood):
    n=neighbourhood*2+1
    matrix=np.zeros((n,n))
    center=np.array([neighbourhood,neighbourhood])
    for i in range(n):
      for j in range(n):
        position=np.array([i,j])
        matrix[i,j]=jump_penalty(center,position)
    return matrix



def video_rgb_to_hsv(video):
  T=video.shape[0]
  for t in range(T):
    video[t,:,:,:]=matplotlib.colors.rgb_to_hsv(video[t,:]/255)


def video_hsv_to_rgb(video):
  T=video.shape[0]
  for t in range(T):
    video[t,:,:,:]=matplotlib.colors.hsv_to_rgb(video[t,:])*255


def draw_tracked(video,track_result):
  T = video.shape[0]
  for t in range(T):
    x,y=tuple(track_result[t,:])
    rr,cc,val=circle_perimeter_aa(x,y,10)
    video[t,rr,cc,:]=255
  return video

def generate_debug_video(video,tracker,debug_video_path):
  FFMpegWriter= anim.writers['ffmpeg']
  metadata=dict(title="Output")
  writer=FFMpegWriter(fps=6,metadata=metadata)
  f,((image_ax,movement_ax),(pixel_ax,score_ax))=plt.subplots(2,2)
  image_ax.set_title("Frame")
  movement_ax.set_title("movement")
  pixel_ax.set_title("pixel")
  score_ax.set_title("score")
  T,h,w,c=video.shape
  with writer.saving(f,debug_video_path,T):
    for t in range(T):
      image_ax.cla()
      pixel_ax.cla()
      score_ax.cla()
      movement_ax.cla()
      plt.suptitle("Frame %d/%d" % (t,T-1))
      image_ax.imshow(video[t,:,:,:])
      pixel_plot=pixel_ax.imshow(tracker.pixel_score[t,:,:])
      f.colorbar(pixel_plot, ax=pixel_ax)
      score_plot=score_ax.imshow(tracker.score[t,:,:])
      f.colorbar(score_plot, ax=score_ax)
      if (t>0):
        movement_plot=movement_ax.imshow(tracker.movement_score[t-1,:,:])
        f.colorbar(movement_plot, ax=movement_ax)
      writer.grab_frame()


if __name__=="__main__":
  # im=io.imread(os.path.join(folderpath,filename))
  test_path='test_data/All_Blacks.5846.main_glosses.mb.r480x360.mp4'
  tracked_video_path= "_tracked".join(os.path.splitext(test_path))
  debug_video_path="_debug".join(os.path.splitext(test_path))
  data = skvideo.io.ffprobe(test_path)['video']
  rate = data['@r_frame_rate']

  video = skvideo.io.vread(test_path).astype(float)
  video=video[1:-1:2,:,:,:]
  video_rgb_to_hsv(video)

  T,h,w,c=video.shape
  neighbourhood_size=15
  movement_vs_pixel_weight=0.7
  jump_penalty_weight=0.0001
  tracker=DPTracker(neighbourhood_size)
  track_result=tracker.track(video,manhattan_movement_score,movement_vs_pixel_weight,skin_pixel_scorer,
                             jump_penalty_euclidean,jump_penalty_weight)

  print(track_result)

  video_hsv_to_rgb(video)
  tracked_video=draw_tracked(video,track_result)
  tracked_video=tracked_video.astype('short')
  generate_debug_video(tracked_video, tracker, debug_video_path)

  skvideo.io.vwrite(tracked_video_path, tracked_video, outputdict={
      '-vcodec': 'libx264',
      '-pix_fmt': 'yuv420p',
      '-r': rate,
  }
                    )
  # for t in range(T):
  #   plt.imshow(video[t,:,:,:])
  #   plt.pause(1)
