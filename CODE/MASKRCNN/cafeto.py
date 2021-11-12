import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")



# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# define a configuration for the model
class CafetoConfig(Config):
	# Give the configuration a recognizable name
	NAME = "cafetos_cfg"
	# Number of classes (background + kangaroo)
	NUM_CLASSES = 3 + 1
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 131

 
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
train_set.load_dataset("/home/jcrivera/Mask_RCNN/samples/cafeto/", is_train=True)
train_set.prepare() 

print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = CafetoDataset()
test_set.load_dataset('/home/jcrivera/Mask_RCNN/samples/cafeto/', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = CafetoConfig()
config.display()

# define the model
model =  modellib.MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights('/home/jcrivera/Mask_RCNN/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


