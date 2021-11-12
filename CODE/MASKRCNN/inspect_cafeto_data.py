import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/jcrivera/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
# import nucleus




# define a configuration for the model
class CafetoConfig(Config):
	# Give the configuration a recognizable name
	NAME = "experimento_maskrcnn_1"
	# Number of classes (background + kangaroo)
	NUM_CLASSES = 4
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 1000
	BACKBONE_STRIDES = [4, 8, 16, 32, 64]
	# RPN_ANCHOR_SCALES = (4,8,16, 32, 64, 128, 256, 512)
	RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
	USE_MINI_MASK = True
	MINI_MASK_SHAPE = (4,4)
	MEAN_PIXEL = np.array([114.7, 127.23, 119.3])
	TRAIN_ROIS_PER_IMAGE = 200
	MASK_SHAPE = [4, 4]
	MAX_GT_INSTANCES = 1000
	DETECTION_MAX_INSTANCES = 1000







 
# prepare config
config = CafetoConfig()

class CafetoDataset(utils.Dataset):

	def load_dataset(self, dataset_dir, is_train= True):
		"""
		"""
		# Add classes. We have one class.
		# Naming the dataset nucleus, and the class nucleus
		self.add_class("dataset", 1, "cafe_verde")
		self.add_class("dataset", 2, "cafe_rojo")
		self.add_class("dataset", 3, "cafe_negro")
		
		# define data locations
		if is_train == True:
			images_dir = dataset_dir + 'train/images/'
			annotations_dir = dataset_dir + 'train/annots/'
		
		else: 
			images_dir = dataset_dir + 'val/images/'
			annotations_dir = dataset_dir + 'val/annots/'
		
		
		# find all images
		for filename in listdir(images_dir):
			image_id = filename[:-4]
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [1,2,3])
		
	
	# load all bounding boxes for an image
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes = list()
		# extract each bounding box
		for box in root.findall('.//object'):

			name = box.find('name').text
			xmin = int(box.find('./bndbox/xmin').text)
			ymin = int(box.find('./bndbox/ymin').text)
			xmax = int(box.find('./bndbox/xmax').text)
			ymax = int(box.find('./bndbox/ymax').text)
			coors = [xmin, ymin, xmax, ymax,name]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	
		
		# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			if box[4]=='cafe_verde':
				masks[row_s:row_e, col_s:col_e, i] = 1
				class_ids.append(self.class_names.index('cafe_verde')) 
			elif box[4]=='cafe_rojo':
				masks[row_s:row_e, col_s:col_e, i] = 2
				class_ids.append(self.class_names.index('cafe_rojo'))
			else:
				masks[row_s:row_e, col_s:col_e, i] = 3
				class_ids.append(self.class_names.index('cafe_negro'))
		return masks, asarray(class_ids, dtype='int32')			
		


	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		return info['path']
        
        
        
 
# train set
train_set = CafetoDataset()
train_set.load_dataset("/home/jcrivera/Mask_RCNN/samples/experimento_1/", is_train=True)
train_set.prepare() 

print("Image Count: {}".format(len(train_set.image_ids)))
print("Class Count: {}".format(train_set.num_classes))


for i, info in enumerate(train_set.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# image_ids = np.random.choice(train_set.image_ids, 4)
# for image_id in image_ids:
#     image = train_set.load_image(image_id)
#     mask, class_ids = train_set.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, train_set.class_names)


# # Load random image and mask.
# image_id = random.choice(train_set.image_ids)
# image = train_set.load_image(image_id)
# mask, class_ids = train_set.load_mask(image_id)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)

# # Display image and additional stats
# print("image_id ", image_id, train_set.image_reference(image_id))
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, train_set.class_names)


# # Load random image and mask.
# image_id = np.random.choice(train_set.image_ids, 1)[0]
# image = train_set.load_image(image_id)
# mask, class_ids = train_set.load_mask(image_id)
# original_shape = image.shape
# # Resize
# image, window, scale, padding, _ = utils.resize_image(
#     image, 
#     min_dim=config.IMAGE_MIN_DIM, 
#     max_dim=config.IMAGE_MAX_DIM,
#     mode=config.IMAGE_RESIZE_MODE)
# mask = utils.resize_mask(mask, scale, padding)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)

# # Display image and additional stats
# print("image_id: ", image_id, train_set.image_reference(image_id))
# print("Original shape: ", original_shape)
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids,train_set.class_names)




# image_id = np.random.choice(train_set.image_ids, 1)[0]
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     train_set, config, image_id, use_mini_mask=False)

# log("image", image)
# log("image_meta", image_meta)
# log("class_ids", class_ids)
# log("bbox", bbox)
# log("mask", mask)

# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])


# # Add augmentation and mask resizing.
# image_id = np.random.choice(train_set.image_ids, 1)[0]
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     train_set, config, image_id, augment=True, use_mini_mask=True)
# log("mask", mask)
# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])



# Generate Anchors
backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
                                          config.RPN_ANCHOR_RATIOS,
                                          backbone_shapes,
                                          config.BACKBONE_STRIDES, 
                                          config.RPN_ANCHOR_STRIDE)

# Print summary of anchors
num_levels = len(backbone_shapes)
anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
print("Count: ", anchors.shape[0])
print("Scales: ", config.RPN_ANCHOR_SCALES)
print("ratios: ", config.RPN_ANCHOR_RATIOS)
print("Anchors per Cell: ", anchors_per_cell)
print("Levels: ", num_levels)
anchors_per_level = []
for l in range(num_levels):
    num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
    anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
    print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))





## Visualize anchors of one cell at the center of the feature map of a specific level

# Load and draw random image

# image_id = np.random.choice(train_set.image_ids, 1)[0]
# image, image_meta, _, _, _ = modellib.load_image_gt(train_set, config, image_id)
# fig, ax = plt.subplots(1, figsize=(10, 10))
# ax.imshow(image)

# levels = len(backbone_shapes)

# for level in range(levels):
#     colors = visualize.random_colors(levels)
#     # Compute the index of the anchors at the center of the image
#     level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
#     level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
#     print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0], 
#                                                                   backbone_shapes[level]))
#     center_cell = backbone_shapes[level] // 2
#     center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
#     level_center = center_cell_index * anchors_per_cell 
#     center_anchor = anchors_per_cell * (
#         (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE**2) \
#         + center_cell[1] / config.RPN_ANCHOR_STRIDE)
#     level_center = int(center_anchor)

#     # Draw anchors. Brightness show the order in the array, dark to bright.
#     for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
#         y1, x1, y2, x2 = rect
#         p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
#                               edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
#         ax.add_patch(p)


# plt.show()



# Create data generator
random_rois = 2000
g = modellib.data_generator(
    train_set, config, shuffle=True, random_rois=random_rois, 
    batch_size=4,
    detection_targets=True)

if random_rois:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
    
    log("rois", rois)
    log("mrcnn_class_ids", mrcnn_class_ids)
    log("mrcnn_bbox", mrcnn_bbox)
    log("mrcnn_mask", mrcnn_mask)
else:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)
    
log("gt_class_ids", gt_class_ids)
log("gt_boxes", gt_boxes)
log("gt_masks", gt_masks)
log("rpn_match", rpn_match, )
log("rpn_bbox", rpn_bbox)
image_id = modellib.parse_image_meta(image_meta)["image_id"][0]
print("image_id: ", image_id, train_set.image_reference(image_id))

# Remove the last dim in mrcnn_class_ids. It's only added
# to satisfy Keras restriction on target shape.
mrcnn_class_ids = mrcnn_class_ids[:,:,0]



b = 0

# Restore original image (reverse normalization)
sample_image = modellib.unmold_image(normalized_images[b], config)

# Compute anchor shifts.
indices = np.where(rpn_match[b] == 1)[0]
refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
log("anchors", anchors)
log("refined_anchors", refined_anchors)

# Get list of positive anchors
positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
print("Positive anchors: {}".format(len(positive_anchor_ids)))
negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
print("Negative anchors: {}".format(len(negative_anchor_ids)))
neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

# ROI breakdown by class
for c, n in zip(train_set.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
    if n:
        print("{:23}: {}".format(c[:20], n))

# Show positive anchors
visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids], 
                     refined_boxes=refined_anchors)










# get_ipython().run_line_magic('matplotlib', 'inline')


# # In[2]:


# # Comment out to reload imported modules if they change
# # %load_ext autoreload
# # %autoreload 2


# # ## Configurations

# # In[3]:


# # Dataset directory
# DATASET_DIR = os.path.join(ROOT_DIR, "datasets/nucleus")

# # Use configuation from nucleus.py, but override
# # image resizing so we see the real sizes here
# class NoResizeConfig(nucleus.NucleusConfig):
#     IMAGE_RESIZE_MODE = "none"
    
# config = NoResizeConfig()


# # ## Notebook Preferences

# # In[4]:


# def get_ax(rows=1, cols=1, size=16):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.
    
#     Adjust the size attribute to control how big to render images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax


# # ## Dataset
# # 
# # Download the dataset from the competition Website. Unzip it and save it in `mask_rcnn/datasets/nucleus`. If you prefer a different directory then change the `DATASET_DIR` variable above.
# # 
# # https://www.kaggle.com/c/data-science-bowl-2018/data

# # In[5]:


# # Load dataset
# dataset = nucleus.NucleusDataset()
# # The subset is the name of the sub-directory, such as stage1_train,
# # stage1_test, ...etc. You can also use these special values:
# #     train: loads stage1_train but excludes validation images
# #     val: loads validation images from stage1_train. For a list
# #          of validation images see nucleus.py
# dataset.load_nucleus(DATASET_DIR, subset="train")

# # Must call before using the dataset
# dataset.prepare()

# print("Image Count: {}".format(len(dataset.image_ids)))
# print("Class Count: {}".format(dataset.num_classes))
# for i, info in enumerate(dataset.class_info):
#     print("{:3}. {:50}".format(i, info['name']))


# # ## Display Samples

# # In[6]:


# # Load and display random samples
# image_ids = np.random.choice(dataset.image_ids, 4)
# for image_id in image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)


# # In[7]:


# # Example of loading a specific image by its source ID
# source_id = "ed5be4b63e9506ad64660dd92a098ffcc0325195298c13c815a73773f1efc279"

# # Map source ID to Dataset image_id
# # Notice the nucleus prefix: it's the name given to the dataset in NucleusDataset
# image_id = dataset.image_from_source_map["nucleus.{}".format(source_id)]

# # Load and display
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#         dataset, config, image_id, use_mini_mask=False)
# log("molded_image", image)
# log("mask", mask)
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names,
#                             show_bbox=False)


# # ## Dataset Stats
# # 
# # Loop through all images in the dataset and collect aggregate stats.

# # In[8]:


# def image_stats(image_id):
#     """Returns a dict of stats for one image."""
#     image = dataset.load_image(image_id)
#     mask, _ = dataset.load_mask(image_id)
#     bbox = utils.extract_bboxes(mask)
#     # Sanity check
#     assert mask.shape[:2] == image.shape[:2]
#     # Return stats dict
#     return {
#         "id": image_id,
#         "shape": list(image.shape),
#         "bbox": [[b[2] - b[0], b[3] - b[1]]
#                  for b in bbox
#                  # Uncomment to exclude nuclei with 1 pixel width
#                  # or height (often on edges)
#                  # if b[2] - b[0] > 1 and b[3] - b[1] > 1
#                 ],
#         "color": np.mean(image, axis=(0, 1)),
#     }

# # Loop through the dataset and compute stats over multiple threads
# # This might take a few minutes
# t_start = time.time()
# with concurrent.futures.ThreadPoolExecutor() as e:
#     stats = list(e.map(image_stats, dataset.image_ids))
# t_total = time.time() - t_start
# print("Total time: {:.1f} seconds".format(t_total))


# # ### Image Size Stats

# # In[9]:


# # Image stats
# image_shape = np.array([s['shape'] for s in stats])
# image_color = np.array([s['color'] for s in stats])
# print("Image Count: ", image_shape.shape[0])
# print("Height  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
#     np.mean(image_shape[:, 0]), np.median(image_shape[:, 0]),
#     np.min(image_shape[:, 0]), np.max(image_shape[:, 0])))
# print("Width   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
#     np.mean(image_shape[:, 1]), np.median(image_shape[:, 1]),
#     np.min(image_shape[:, 1]), np.max(image_shape[:, 1])))
# print("Color   mean (RGB): {:.2f} {:.2f} {:.2f}".format(*np.mean(image_color, axis=0)))

# # Histograms
# fig, ax = plt.subplots(1, 3, figsize=(16, 4))
# ax[0].set_title("Height")
# _ = ax[0].hist(image_shape[:, 0], bins=20)
# ax[1].set_title("Width")
# _ = ax[1].hist(image_shape[:, 1], bins=20)
# ax[2].set_title("Height & Width")
# _ = ax[2].hist2d(image_shape[:, 1], image_shape[:, 0], bins=10, cmap="Blues")


# # ### Nuclei per Image Stats

# # In[10]:


# # Segment by image area
# image_area_bins = [256**2, 600**2, 1300**2]

# print("Nuclei/Image")
# fig, ax = plt.subplots(1, len(image_area_bins), figsize=(16, 4))
# area_threshold = 0
# for i, image_area in enumerate(image_area_bins):
#     nuclei_per_image = np.array([len(s['bbox']) 
#                                  for s in stats 
#                                  if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area])
#     area_threshold = image_area
#     if len(nuclei_per_image) == 0:
#         print("Image area <= {:4}**2: None".format(np.sqrt(image_area)))
#         continue
#     print("Image area <= {:4.0f}**2:  mean: {:.1f}  median: {:.1f}  min: {:.1f}  max: {:.1f}".format(
#         np.sqrt(image_area), nuclei_per_image.mean(), np.median(nuclei_per_image), 
#         nuclei_per_image.min(), nuclei_per_image.max()))
#     ax[i].set_title("Image Area <= {:4}**2".format(np.sqrt(image_area)))
#     _ = ax[i].hist(nuclei_per_image, bins=10)


# # ### Nuclei Size Stats

# # In[11]:


# # Nuclei size stats
# fig, ax = plt.subplots(1, len(image_area_bins), figsize=(16, 4))
# area_threshold = 0
# for i, image_area in enumerate(image_area_bins):
#     nucleus_shape = np.array([
#         b 
#         for s in stats if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area
#         for b in s['bbox']])
#     nucleus_area = nucleus_shape[:, 0] * nucleus_shape[:, 1]
#     area_threshold = image_area

#     print("\nImage Area <= {:.0f}**2".format(np.sqrt(image_area)))
#     print("  Total Nuclei: ", nucleus_shape.shape[0])
#     print("  Nucleus Height. mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
#         np.mean(nucleus_shape[:, 0]), np.median(nucleus_shape[:, 0]),
#         np.min(nucleus_shape[:, 0]), np.max(nucleus_shape[:, 0])))
#     print("  Nucleus Width.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
#         np.mean(nucleus_shape[:, 1]), np.median(nucleus_shape[:, 1]),
#         np.min(nucleus_shape[:, 1]), np.max(nucleus_shape[:, 1])))
#     print("  Nucleus Area.   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
#         np.mean(nucleus_area), np.median(nucleus_area),
#         np.min(nucleus_area), np.max(nucleus_area)))

#     # Show 2D histogram
#     _ = ax[i].hist2d(nucleus_shape[:, 1], nucleus_shape[:, 0], bins=20, cmap="Blues")


# # In[12]:


# # Nuclei height/width ratio
# nucleus_aspect_ratio = nucleus_shape[:, 0] / nucleus_shape[:, 1]
# print("Nucleus Aspect Ratio.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
#     np.mean(nucleus_aspect_ratio), np.median(nucleus_aspect_ratio),
#     np.min(nucleus_aspect_ratio), np.max(nucleus_aspect_ratio)))
# plt.figure(figsize=(15, 5))
# _ = plt.hist(nucleus_aspect_ratio, bins=100, range=[0, 5])


# # ## Image Augmentation
# # 
# # Test out different augmentation methods

# # In[13]:


# # List of augmentations
# # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
# augmentation = iaa.Sometimes(0.9, [
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
#     iaa.Multiply((0.8, 1.2)),
#     iaa.GaussianBlur(sigma=(0.0, 5.0))
# ])


# # In[14]:


# # Load the image multiple times to show augmentations
# limit = 4
# ax = get_ax(rows=2, cols=limit//2)
# for i in range(limit):
#     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#         dataset, config, image_id, use_mini_mask=False, augment=False, augmentation=augmentation)
#     visualize.display_instances(image, bbox, mask, class_ids,
#                                 dataset.class_names, ax=ax[i//2, i % 2],
#                                 show_mask=False, show_bbox=False)


# # ## Image Crops
# # 
# # Microscoy images tend to be large, but nuclei are small. So it's more efficient to train on random crops from large images. This is handled by `config.IMAGE_RESIZE_MODE = "crop"`.
# # 
# # 

# # In[15]:


# class RandomCropConfig(nucleus.NucleusConfig):
#     IMAGE_RESIZE_MODE = "crop"
#     IMAGE_MIN_DIM = 256
#     IMAGE_MAX_DIM = 256

# crop_config = RandomCropConfig()


# # In[16]:


# # Load the image multiple times to show augmentations
# limit = 4
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# ax = get_ax(rows=2, cols=limit//2)
# for i in range(limit):
#     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#         dataset, crop_config, image_id, use_mini_mask=False)
#     visualize.display_instances(image, bbox, mask, class_ids,
#                                 dataset.class_names, ax=ax[i//2, i % 2],
#                                 show_mask=False, show_bbox=False)


# # ## Mini Masks
# # 
# # Instance binary masks can get large when training with high resolution images. For example, if training with 1024x1024 image then the mask of a single instance requires 1MB of memory (Numpy uses bytes for boolean values). If an image has 100 instances then that's 100MB for the masks alone. 
# # 
# # To improve training speed, we optimize masks:
# # * We store mask pixels that are inside the object bounding box, rather than a mask of the full image. Most objects are small compared to the image size, so we save space by not storing a lot of zeros around the object.
# # * We resize the mask to a smaller size (e.g. 56x56). For objects that are larger than the selected size we lose a bit of accuracy. But most object annotations are not very accuracy to begin with, so this loss is negligable for most practical purposes. Thie size of the mini_mask can be set in the config class.
# # 
# # To visualize the effect of mask resizing, and to verify the code correctness, we visualize some examples.

# # In[17]:


# # Load random image and mask.
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image = dataset.load_image(image_id)
# mask, class_ids = dataset.load_mask(image_id)
# original_shape = image.shape
# # Resize
# image, window, scale, padding, _ = utils.resize_image(
#     image, 
#     min_dim=config.IMAGE_MIN_DIM, 
#     max_dim=config.IMAGE_MAX_DIM,
#     mode=config.IMAGE_RESIZE_MODE)
# mask = utils.resize_mask(mask, scale, padding)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)

# # Display image and additional stats
# print("image_id: ", image_id, dataset.image_reference(image_id))
# print("Original shape: ", original_shape)
# log("image", image)
# log("mask", mask)
# log("class_ids", class_ids)
# log("bbox", bbox)
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# # In[18]:


# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset, config, image_id, use_mini_mask=False)

# log("image", image)
# log("image_meta", image_meta)
# log("class_ids", class_ids)
# log("bbox", bbox)
# log("mask", mask)

# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])


# # In[19]:


# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# # In[20]:


# # Add augmentation and mask resizing.
# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#     dataset, config, image_id, augment=True, use_mini_mask=True)
# log("mask", mask)
# display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])


# # In[21]:


# mask = utils.expand_mask(bbox, mask, image.shape)
# visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# # ## Anchors
# # 
# # For an FPN network, the anchors must be ordered in a way that makes it easy to match anchors to the output of the convolution layers that predict anchor scores and shifts. 
# # * Sort by pyramid level first. All anchors of the first level, then all of the second and so on. This makes it easier to separate anchors by level.
# # * Within each level, sort anchors by feature map processing sequence. Typically, a convolution layer processes a feature map starting from top-left and moving right row by row. 
# # * For each feature map cell, pick any sorting order for the anchors of different ratios. Here we match the order of ratios passed to the function.

# # In[22]:


# ## Visualize anchors of one cell at the center of the feature map

# # Load and display random image
# image_id = np.random.choice(dataset.image_ids, 1)[0]
# image, image_meta, _, _, _ = modellib.load_image_gt(dataset, crop_config, image_id)

# # Generate Anchors
# backbone_shapes = modellib.compute_backbone_shapes(config, image.shape)
# anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
#                                           config.RPN_ANCHOR_RATIOS,
#                                           backbone_shapes,
#                                           config.BACKBONE_STRIDES, 
#                                           config.RPN_ANCHOR_STRIDE)

# # Print summary of anchors
# num_levels = len(backbone_shapes)
# anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
# print("Count: ", anchors.shape[0])
# print("Scales: ", config.RPN_ANCHOR_SCALES)
# print("ratios: ", config.RPN_ANCHOR_RATIOS)
# print("Anchors per Cell: ", anchors_per_cell)
# print("Levels: ", num_levels)
# anchors_per_level = []
# for l in range(num_levels):
#     num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
#     anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
#     print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))

# # Display
# fig, ax = plt.subplots(1, figsize=(10, 10))
# ax.imshow(image)
# levels = len(backbone_shapes)

# for level in range(levels):
#     colors = visualize.random_colors(levels)
#     # Compute the index of the anchors at the center of the image
#     level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
#     level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
#     print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0], 
#                                                                 backbone_shapes[level]))
#     center_cell = backbone_shapes[level] // 2
#     center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
#     level_center = center_cell_index * anchors_per_cell 
#     center_anchor = anchors_per_cell * (
#         (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE**2) \
#         + center_cell[1] / config.RPN_ANCHOR_STRIDE)
#     level_center = int(center_anchor)

#     # Draw anchors. Brightness show the order in the array, dark to bright.
#     for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
#         y1, x1, y2, x2 = rect
#         p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
#                               edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
#         ax.add_patch(p)


# # ## Data Generator

# # In[23]:


# # Create data generator
# random_rois = 2000
# g = modellib.data_generator(
#     dataset, crop_config, shuffle=True, random_rois=random_rois, 
#     batch_size=4,
#     detection_targets=True)


# # In[24]:


# # Uncomment to run the generator through a lot of images
# # to catch rare errors
# # for i in range(1000):
# #     print(i)
# #     _, _ = next(g)


# # In[25]:


# # Get Next Image
# if random_rois:
#     [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois],     [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
    
#     log("rois", rois)
#     log("mrcnn_class_ids", mrcnn_class_ids)
#     log("mrcnn_bbox", mrcnn_bbox)
#     log("mrcnn_mask", mrcnn_mask)
# else:
#     [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)
    
# log("gt_class_ids", gt_class_ids)
# log("gt_boxes", gt_boxes)
# log("gt_masks", gt_masks)
# log("rpn_match", rpn_match, )
# log("rpn_bbox", rpn_bbox)
# image_id = modellib.parse_image_meta(image_meta)["image_id"][0]
# print("image_id: ", image_id, dataset.image_reference(image_id))

# # Remove the last dim in mrcnn_class_ids. It's only added
# # to satisfy Keras restriction on target shape.
# mrcnn_class_ids = mrcnn_class_ids[:,:,0]


# # In[26]:


# b = 0

# # Restore original image (reverse normalization)
# sample_image = modellib.unmold_image(normalized_images[b], config)

# # Compute anchor shifts.
# indices = np.where(rpn_match[b] == 1)[0]
# refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
# log("anchors", anchors)
# log("refined_anchors", refined_anchors)

# # Get list of positive anchors
# positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
# print("Positive anchors: {}".format(len(positive_anchor_ids)))
# negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
# print("Negative anchors: {}".format(len(negative_anchor_ids)))
# neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
# print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

# # ROI breakdown by class
# for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
#     if n:
#         print("{:23}: {}".format(c[:20], n))

# # Show positive anchors
# fig, ax = plt.subplots(1, figsize=(16, 16))
# visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids], 
#                      refined_boxes=refined_anchors, ax=ax)


# # In[27]:


# # Show negative anchors
# visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])


# # In[28]:


# # Show neutral anchors. They don't contribute to training.
# visualize.draw_boxes(sample_image, boxes=anchors[np.random.choice(neutral_anchor_ids, 100)])


# # ## ROIs
# # 
# # Typically, the RPN network generates region proposals (a.k.a. Regions of Interest, or ROIs). The data generator has the ability to generate proposals as well for illustration and testing purposes. These are controlled by the `random_rois` parameter.

# # In[29]:


# if random_rois:
#     # Class aware bboxes
#     bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]

#     # Refined ROIs
#     refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:,:4] * config.BBOX_STD_DEV)

#     # Class aware masks
#     mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]

#     visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)
    
#     # Any repeated ROIs?
#     rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
#     _, idx = np.unique(rows, return_index=True)
#     print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))


# # In[30]:


# if random_rois:
#     # Dispalay ROIs and corresponding masks and bounding boxes
#     ids = random.sample(range(rois.shape[1]), 8)

#     images = []
#     titles = []
#     for i in ids:
#         image = visualize.draw_box(sample_image.copy(), rois[b,i,:4].astype(np.int32), [255, 0, 0])
#         image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
#         images.append(image)
#         titles.append("ROI {}".format(i))
#         images.append(mask_specific[i] * 255)
#         titles.append(dataset.class_names[mrcnn_class_ids[b,i]][:20])

#     display_images(images, titles, cols=4, cmap="Blues", interpolation="none")


# # In[31]:


# # Check ratio of positive ROIs in a set of images.
# if random_rois:
#     limit = 10
#     temp_g = modellib.data_generator(
#         dataset, crop_config, shuffle=True, random_rois=10000, 
#         batch_size=1, detection_targets=True)
#     total = 0
#     for i in range(limit):
#         _, [ids, _, _] = next(temp_g)
#         positive_rois = np.sum(ids[0] > 0)
#         total += positive_rois
#         print("{:5} {:5.2f}".format(positive_rois, positive_rois/ids.shape[1]))
#     print("Average percent: {:.2f}".format(total/(limit*ids.shape[1])))


# # In[ ]:




