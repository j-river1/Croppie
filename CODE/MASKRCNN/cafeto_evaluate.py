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
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap
from mrcnn.model import mold_image


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils



# define a configuration for the model
class PredictionConfig(Config):
	# Give the configuration a recognizable name
	NAME = "cafetos_cfg"
	# Number of classes (background + kangaroo)
	NUM_CLASSES = 3 + 1


	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	USE_MINI_MASK= False

 


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
        
 # calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)

		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		print(type(scaled_image))
		print(scaled_image.shape)		
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		print(type(sample))
		print(sample.shape)
		# make prediction
		yhat = model.detect(sample, verbose=1)
		# print(type(yhat))
		# print(len(yhat))
		# print((type(yhat[0])))
		# # print(yhat[0]["class_ids"])
		# print(yhat[0]["scores"])
		# print(yhat[0]['masks'])
		

		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		AP =np.nan_to_num(AP)
		print(AP)
		print(image_id)
		print(dataset.source_image_link(image_id))
		# store
		APs.append(AP)
	# # calculate the mean AP across all images
	mAP = mean(APs)
	return mAP       
        
 
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
config = PredictionConfig()

# define the model
model = modellib.MaskRCNN(mode='inference', model_dir='./', config=config)
# load model weights
model.load_weights("/home/jcrivera/Mask_RCNN/cafetos_cfg20210916T1918/mask_rcnn_cafetos_cfg_0004.h5", by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, config)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, config)
print("Test mAP: %.3f" % test_mAP)