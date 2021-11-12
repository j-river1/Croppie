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
# from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import pandas as pd
matplotlib.use('Agg')



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
# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, guardar, model, cfg, n_images=1):
	# load image and mask
	# for i in range(n_images):


	lista = []
	for filename in os.listdir(dataset):
		filename_= os.path.join(dataset, filename)
		# print(filename_)
		image = skimage.io.imread(filename_)
		# If grayscale. Convert to RGB for consistency.
		if image.ndim != 3:
			image = skimage.color.gray2rgb(image)
		# If has an alpha channel, remove it for consistency
		if image.shape[-1] == 4:
			image = image[..., :3]


	# 	# load the image and mask
		#image = dataset.load_image(i+10)
		


	# 	# mask, _ = dataset.load_mask(i+10)
	# 	# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
	# 	# convert image into one sample
		sample = expand_dims(scaled_image, 0)
	# 	# make prediction
		yhat = model.detect(sample, verbose=0)[0]
	# 	# define subplot real
		fig, (axs1, axs2) = plt.subplots(1, 2)
		axs1.imshow(image)
		# fig.suptitle('Actual')
		axs1.set_title('Actual')
		axs2.imshow(image)
		ax = plt.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
		
		lista.append((filename[:-4], len(yhat['rois'])))
		# print(len(yhat['rois']))
		axs2.set_title('Predict: ' + str(len(yhat['rois'])))
		plt.gcf().set_size_inches(15, 15)
		nombre =  guardar +filename[:-4] +"_predi.png"
		print(nombre)
		
		#print(lista)
		# plt.savefig(nombre, dpi=600, transparent=True) 
		plt.savefig(nombre, transparent=True) 
		plt.close(fig)
		# axs2.imshow(image)
	print(lista)

	df = pd.DataFrame(lista, columns =['Foto', 'CantidadGranos'])
	nombre_archivo = guardar + "resumen.csv"
	df.to_csv(nombre_archivo, index=False)

	print(df)


	# 	pyplot.subplot(n_images, 2, i*2+1)
	# 	# plot raw pixel data
	# 	pyplot.imshow(image)
	# 	pyplot.title('Actual')
	# 	# plot masks
	# 	# for j in range(mask.shape[2]):
	# 	# 	pyplot.imshow(mask[:, :, j], alpha=0)
	# 	# get the context for drawing boxes
	# 	pyplot.subplot(n_images, 2, i*2+2)
	# 	# plot raw pixel data
	# 	pyplot.imshow(image)
	# 	pyplot.title('Predicted')
	# 	ax = pyplot.gca()
	# 	# plot each box
	# 	for box in yhat['rois']:
	# 		# get coordinates
	# 		y1, x1, y2, x2 = box
	# 		# calculate width and height of the box
	# 		width, height = x2 - x1, y2 - y1
	# 		# create the shape
	# 		rect = Rectangle((x1, y1), width, height, fill=False, color='red')
	# 		# draw the box
	# 		ax.add_patch(rect)
	# # # show the figur
	# # # pyplot.show()      
		# plt.gcf().set_size_inches(15, 15)
	# plt.savefig("comments_2.png", dpi=600)  
	# nombre =  filename[:-4] +".png"
	# print(nombre)
	# plt.savefig(nombre, dpi=600, transparent=True)   
	   
 
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

# predict = CafetoDataset()
# predict_ = predict.add_image('dataset_', image_id="1615328348663.jpg", path="/home/jcrivera/Mask_RCNN/samples/cafeto/val_1/")


# plot predictions for train dataset
# plot_actual_vs_predicted(train_set, model,config)
# plot predictions for test dataset
# plot_actual_vs_predicted(test_set, model, config)

# plot_actual_vs_predicted("/home/jcrivera/Mask_RCNN/samples/cafeto/val_1/1615328348663.jpg", model, config)


plot_actual_vs_predicted("/home/jcrivera/Mask_RCNN/samples/cafeto/Contar_/","/home/jcrivera/Mask_RCNN/samples/cafeto/Contar_predic/", model, config)




# plot_actual_vs_predicted(predict_, model, config)
