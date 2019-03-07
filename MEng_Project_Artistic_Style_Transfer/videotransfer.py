import numpy as np
import cv2
from PIL import Image
import os
import glob
from natsort import natsorted
from moviepy.editor import *
from styletransfer import style_transfer
from moviepy.editor import VideoFileClip

vidcap = cv2.VideoCapture("me3.mp4")
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite("content/me3/%d.jpg" % count, image)     # save frame as JPEG file
  count += 1


style_images_paths = 'style7.jpeg'
content_images_paths = 'content/me3/me3255/*.jpg'
checkpoints_path = 'model.ckpt'
output_dir = 'output/me3/'
image_size = 255	
style_image_size = 255
maximum_styles_to_evaluate = 1024
content_square_crop = False
style_square_crop = False
interpolation = '[0.8]'


style_images_paths = 'style7.jpeg'
content_images_paths = 'content/me3/me3255/*.jpg'
checkpoints_path = 'model.ckpt'
output_dir = 'output/me3/'
image_size = 255	
style_image_size = 255
maximum_styles_to_evaluate = 1024
content_square_crop = False
style_square_crop = False
interpolation = '[0.8]'

style_transfer(		style_images_paths,
			content_images_paths,
			checkpoints_path,
			output_dir,
			image_size,
			style_image_size,
			maximum_styles_to_evaluate,
			content_square_crop,
			style_square_crop,
			interpolation)


base_dir = os.path.realpath("output/me2/style7/")
print(base_dir)


file_list = glob.glob('output/me3/*.jpg')  # Get all the pngs in the current directory
file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
# 0.04 290
clips = [ImageClip(m).set_duration(0.03)
         for m in file_list_sorted]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("style7me3.mp4", fps=25)
