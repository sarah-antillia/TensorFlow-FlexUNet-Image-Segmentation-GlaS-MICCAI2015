# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/06/27 ImageMaskDatasetGenerator.py

import os
import glob
import cv2
import numpy as np

import csv
import traceback
import shutil

from PIL import Image, ImageOps
import traceback
import math
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class ImageMaskDatasetGenerator:

  def __init__(self, csv_file= "./Warwick_QU_Dataset/Grade.csv", augmentation=False):
    self.H    = 512
    self.W    = 512
    self.seed = 137
    self.file_format = ".bmp"
    self.out_format  = ".png"
    # BGR color_map
    self.bgr_color_map = {}
    self.bgr_color_map["benign"]    = (0, 255, 0)  #green
    self.bgr_color_map["malignant"] = (0, 0, 255)  #red

    with open(csv_file, newline='', encoding='utf-8') as csvfile:
      reader = csv.reader(csvfile)
      rows = list(reader) 

      # Remove the header from rows.
      rows = rows[1:]
      i = 1
      # Create a map of filename and pathology from the csv_file.
      self.filename_pathology = {}
      for row in rows:
         filename   = row[0].strip()
         pathology  = row[2].strip()
         self.filename_pathology[filename ] = pathology
         #print("No {} filename {} pathologh {}".format(i, filename,pathology))
         i += 1
      #print(self.filename_pathology)

    self.augmentation = augmentation

    if self.augmentation:
      self.hflip    = False
      self.vflip    = False
      self.rotation = False
      #self.ANGLES   = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
      self.ANGLES   = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
      self.ANGLES   = [90, 180, 270, ]

      self.deformation=True
      self.alpha    = 1300
      self.sigmoids = [8, 9, 10,]
          
      self.distortion=True
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5
      self.distortions           = [0.02, 0.03, 0.04]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)
      
      self.resize = False
      self.barrel_distortion = True
      self.radius     = 0.3
      self.amounts    = [0.3]
      self.centers    = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

      self.pincushion_distortion= True
      self.pincradius  = 0.3
      self.pincamounts = [-0.3]
      self.pinccenters = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]


  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
    #image_files = glob.glob(images_dir + "/*" + self.file_format)
    mask_files  = glob.glob(masks_dir  + "/*_*_anno" + self.file_format)
    for mask_file in mask_files:
      basename  = os.path.basename(mask_file)
      name_with_anno = basename.split(".")[0]
      name      = name_with_anno.replace("_anno", "")
      image_file = mask_file.replace("_anno", "")
      pathology = self.filename_pathology[name]
      output_filename = pathology + "_" + name  + self.out_format
      output_filepath = os.path.join(output_images_dir, output_filename)
      image   = cv2.imread(image_file)
      image  = self.resize_to_square(image, mask=False)  
    
      cv2.imwrite(output_filepath, image)
      print("Saved {}".format(output_filepath))
      if self.augmentation:
        self.augment(image, output_filename, output_images_dir, border=(0, 0, 0), mask=False)

    for mask_file in mask_files:
      basename  = os.path.basename(mask_file)
      name_with_anno = basename.split(".")[0]
      name      = name_with_anno.replace("_anno", "")
      pathology = self.filename_pathology[name]
      output_filename = pathology + "_" + name  + self.out_format
      output_filepath = os.path.join(output_masks_dir, output_filename)
      mask   = cv2.imread(mask_file)
      mask   = mask* 255
      mask  = self.resize_to_square(mask, mask=True)  
      mask  = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      color = self.bgr_color_map[pathology]
      mask  = self.colorize_mask(mask, color=color, gray=255)
      cv2.imwrite(output_filepath, mask)
      print("Saved {}".format(output_filepath))
      if self.augmentation:
        self.augment(mask, output_filename, output_masks_dir, border=(0, 0, 0), mask=True)

  def resize_to_square(self, image, mask=True):
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    if mask:
      background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 
    else:
      pixel = image[10][10]
      background = np.ones((RESIZE, RESIZE, 3),  np.uint8) * pixel 
      #background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 

    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (512x512)
    resized = cv2.resize(background, (self.W, self.H))

    return resized
  
  def colorize_mask(self, mask, color=(255, 255, 255), gray=0):
    h, w = mask.shape[:2]
    rgb_mask = np.zeros((w, h, 3), np.uint8)
    #condition = (mask[...] == gray) 
    condition = (mask[...] >= gray-10) & (mask[...] <= gray+10)   
    rgb_mask[condition] = [color]  
    return rgb_mask   


  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    border = image[2][2].tolist()
  
    print("---- border {}".format(border))
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.deformation:
      self.deform(image, basename, output_dir)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.resize:
      self.shrink(image, basename, output_dir, mask)

    if self.barrel_distortion:
      self.barrel_distort(image, basename, output_dir)


    if self.pincushion_distortion:
      self.pincushion_distort(image, basename, output_dir)

  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))

  def deform(self, image, basename, output_dir): 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    print("--- shape {}".format(shape))

    for sigmoid in self.sigmoids:
      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

      deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed_image = deformed_image.reshape(image.shape)

      image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, deformed_image)
      print("=== Saved {}".format(image_filepath))


  # This method is based on the code of the following stackoverflow.com webstie:
  # https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/78031420#78031420
  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved {}".format(output_file))

  def shrink(self, image, basename, output_dir, mask):
    print("----shrink shape {}".format(image.shape))
    h, w    = image.shape[0:2]
    pixel   = image[2][2]
    for resize_ratio in self.resize_ratios:
      rh = int(h * resize_ratio)
      rw = int(w * resize_ratio)
      resized = cv2.resize(image, (rw, rh))
      h1, w1  = resized.shape[:2]
      y = int((h - h1)/2)
      x = int((w - w1)/2)
      # black background
      background = np.zeros((w, h, 3), np.uint8)
      if mask == False:
        # white background
        background = np.ones((h, w, 3), np.uint8) * pixel
      # paste resized to background
      print("---shrink mask {} rsized.shape {}".format(mask, resized.shape))
      background[x:x+w1, y:y+h1] = resized
      filename = "shrinked_" + str(resize_ratio) + "_" + basename
      output_file = os.path.join(output_dir, filename)    

      cv2.imwrite(output_file, background)
      print("=== Saved shrinked image file{}".format(output_file))

  # This method is based on the code in the following stackoverflow.com website:
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion
  def barrel_distort(self, image, basename, output_dir):    
    (h,  w,  _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index   = 1000
    for amount in self.amounts:
      for center in self.centers:
        index += 1
        (ox, oy) = center
        center_x = w * ox
        center_y = h * oy
        radius = w * self.radius
           
        # negative values produce pincushion
 
        # create map with the barrel pincushion distortion formula
        for y in range(h):
          delta_y = scale_y * (y - center_y)
          for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
              map_x[y, x] = x
              map_y[y, x] = y
            else:
              factor = 1.0
              if distance > 0.0:
                v = math.sqrt(distance)
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
              map_x[y, x] = factor * delta_x / scale_x + center_x
              map_y[y, x] = factor * delta_y / scale_y + center_y
            
        # do the remap
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        filename = "barrdistorted_"+str(index) + "_" + str(self.radius) + "_" + str(amount) + "_" + basename
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, image)
        print("Saved {}".format(output_filepath))

  # This method is based on the code in the following stackoverflow.com website:
  # https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion
  def pincushion_distort(self, image, basename, output_dir):    
    (h,  w,  _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index   = 1000
    for amount in self.pincamounts:
      for center in self.pinccenters:
        index += 1
        (ox, oy) = center
        center_x = w * ox
        center_y = h * oy
        radius = w * self.pincradius
           
        # negative values produce pincushion

        # create map with the barrel pincushion distortion formula
        for y in range(h):
          delta_y = scale_y * (y - center_y)
          for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
              map_x[y, x] = x
              map_y[y, x] = y
            else:
              factor = 1.0
              if distance > 0.0:
                v = math.sqrt(distance)
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
              map_x[y, x] = factor * delta_x / scale_x + center_x
              map_y[y, x] = factor * delta_y / scale_y + center_y
            
        # do the remap
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        filename = "pincdistorted_"+str(index) + "_" + str(self.pincradius) + "_" + str(amount) + "_" + basename
        output_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_filepath, image)
        print("Saved {}".format(output_filepath))

if __name__ == "__main__":

  try:
    # Input file and dirs
    csv_file          = "./Warwick_QU_Dataset/Grade.csv"
    images_dir        = "./Warwick_QU_Dataset/"
    masks_dir         = "./Warwick_QU_Dataset/"

    # Output dirs
    output_dir        = "./Glas-master/"
    output_images_dir = "./Glas-master/images"
    output_masks_dir  = "./Glas-master/masks"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    augmentation = True
    generator = ImageMaskDatasetGenerator(csv_file=csv_file, augmentation=augmentation)

    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)
    
  except:
    traceback.print_exc()


