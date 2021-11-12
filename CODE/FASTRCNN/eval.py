#!/usr/bin/env python
# coding: utf-8

# # initial setup

# In[1]:


from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn.pascal_voc import pascal_voc_util
#from keras_frcnn.pascal_voc_parser import get_data
from keras_frcnn.simple_parser import get_data
from keras_frcnn import data_generators

from utils import get_bbox


# In[2]:


config_output_filename = "config.pickle"

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)
C.network = 'vgg16'
from keras_frcnn import vgg as nn


# In[3]:


img_path = "/home/jcrivera/frcnn-from-scratch-with-keras/train/"
# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)
def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}

class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(50)

num_features = 512
print(class_mapping)


# # In[4]:


get_ipython().system(' ls models')


# In[5]:


C.model_path = "models/vgg/voc.hdf5"


# # setup model

# In[6]:


if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)

# model loading
print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)


model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.05

visualise = True


# # # set up dataset

# # In[7]:


# # define pascal
# DEVKIT_PATH = "../VOCdevkit"

# DEVKIT_PATH = "/home/jcrivera/frcnn-from-scratch-with-keras/train/"

DEVKIT_PATH = "/home/jcrivera/frcnn-from-scratch-with-keras/train_labels.txt"
SET = "trainval"
# pascal = pascal_voc_util(DEVKIT_PATH)

# # define dataloader
all_imgs, classes_count, _ = get_data(DEVKIT_PATH)
print(all_imgs)
val_imgs = [s for s in all_imgs if s['imageset'] == SET]
if len(val_imgs) == 0:
	print(SET, " images not found. using trainval images for testing.")
	val_imgs = [s for s in all_imgs if s['imageset'] == 'trainval'] # for test purpose
	
print('Num val samples {}'.format(len(val_imgs)))
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

img_pathes = [x["filepath"] for x in val_imgs]

# define detections
all_boxes = [[[] for _ in range(len(val_imgs))]
			   for _ in range(21)]
empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))


# # In[8]:


# all_imgs


# # In[9]:


len(all_boxes[0])


# # # infer data

# # In[10]:


image_index = sorted(img_pathes)
print(image_index)

for idx, img_name in enumerate(image_index):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print("inference image path:", img_name)
	st = time.time()
	filepath = img_name

	img = cv2.imread(filepath)

	X, ratio = format_img(img, C)
	img_scaled = (np.transpose(X[0,:,:,:],(1,2,0)) + 127.5).astype('uint8')

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	
	# infer roi
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.5)
	# get bbox
#	all_dets, bboxes, probs = get_bbox(R, C, model_classifier, class_mapping, F, ratio, bbox_threshold=0.5)
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold: #or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				print("no boxes detected")
				continue
			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))
	
	for keyid in class_mapping:   
		key = class_mapping[keyid]
		if key not in bboxes or key == "bg":
			all_boxes[keyid][idx] = empty_array
			continue
		else:
			print("detections of ", key)
		if key == "bg":
			continue

		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.45)
		all_dets = []
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
			all_dets.append([real_x1, real_y1, real_x2, real_y2, new_probs[jk]])
		all_boxes[keyid][idx] = all_dets


# # In[11]:


all_boxes[0]


# In[12]:


for i in class_mapping.items():
	print(i)


# # evaluate accuracy

# In[13]:


# eval function
def voc_eval(detpath,
			 annopath,
			 imagesetfile,
			 classname,
			 cachedir,
			 ovthresh=0.5,
			 use_07_metric=False):
  """
  rec, prec, ap = voc_eval(detpath,
							  annopath,
							  imagesetfile,
							  classname,
							  [ovthresh],
							  [use_07_metric])
  Top level function that does the PASCAL VOC evaluation.
  detpath: Path to detections
	  detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
	  annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
	  (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
	os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
	lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
	# load annotations
	recs = {}
	for i, imagename in enumerate(imagenames):
	  recs[imagename] = parse_rec(annopath.format(imagename))
	  if i % 100 == 0:
		print('Reading annotation for {:d}/{:d}'.format(
		  i + 1, len(imagenames)))
	# save
	#print('Saving cached annotations to {:s}'.format(cachefile))
	#with open(cachefile, 'wb') as f:
	#  pickle.dump(recs, f)
  else:
	# load
	with open(cachefile, 'rb') as f:
	  try:
		recs = pickle.load(f)
	  except:
		recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
	R = [obj for obj in recs[imagename] if obj['name'] == classname]
	bbox = np.array([x['bbox'] for x in R])
	difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
	det = [False] * len(R)
	npos = npos + sum(~difficult)
	class_recs[imagename] = {'bbox': bbox,
							 'difficult': difficult,
							 'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
	lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
	# sort by confidence
	sorted_ind = np.argsort(-confidence)
#    sorted_scores = np.sort(-confidence)
	BB = BB[sorted_ind, :]
	image_ids = [image_ids[x] for x in sorted_ind]

	# go down dets and mark TPs and FPs
	for d in range(nd):
	  id = image_ids[d][-10:-4]
	  # catch bad detections
	  try:
		  R = class_recs[id]
	  except:
		print("det not found")
		continue
		
	  bb = BB[d, :].astype(float)
	  ovmax = -np.inf
	  BBGT = R['bbox'].astype(float)

	  if BBGT.size > 0:
		# compute overlaps
		# intersection
		ixmin = np.maximum(BBGT[:, 0], bb[0])
		iymin = np.maximum(BBGT[:, 1], bb[1])
		ixmax = np.minimum(BBGT[:, 2], bb[2])
		iymax = np.minimum(BBGT[:, 3], bb[3])
		iw = np.maximum(ixmax - ixmin + 1., 0.)
		ih = np.maximum(iymax - iymin + 1., 0.)
		inters = iw * ih

		# union
		uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
			   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
			   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

		overlaps = inters / uni
		ovmax = np.max(overlaps)
		jmax = np.argmax(overlaps)

	  if ovmax > ovthresh:
		if not R['difficult'][jmax]:
		  if not R['det'][jmax]:
			tp[d] = 1.
			R['det'][jmax] = 1
		  else:
			fp[d] = 1.
	  else:
		fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap


# # In[14]:


# pascal_classes = np.asarray(['dog', 'cat', 'car', 'person', 'chair', 'bottle', 'diningtable', 'pottedplant', 'bird', 'horse', 'motorbike', 'bus', 'tvmonitor', 'sofa', 'boat', 'cow', 'aeroplane', 'train', 'sheep', 'bicycle', 'bg'])
# PASCAL_CLASSES = pascal_classes

pascal_classes = np.asarray(['cafe_verde', 'cafe_rojo', 'cafe_negro', 'bg'])
PASCAL_CLASSES = pascal_classes



# # In[15]:


pascal_classes


# ## write out detections to evaluate on official script

# In[27]:


def get_voc_results_file_template(cls):
		# VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
		filename = 'det_' + "val" + '_'+cls+'.txt'
		filedir = "/home/jcrivera/frcnn-from-scratch-with-keras/results"
		# filedir = os.path.join(DEVKIT_PATH, 'results', 'VOC2007', 'Main')
		if not os.path.exists(filedir):
			os.makedirs(filedir)
		path = os.path.join(filedir, filename)
		return path


def write_voc_results_file(pascal_classes, all_boxes, image_index):
		for cls_ind, cls in enumerate(pascal_classes):
			if cls == '__background__':
				continue
			print('Writing {} VOC results file'.format(cls))
			filename = get_voc_results_file_template(cls)
			with open(filename, 'wt') as f:
				for im_ind, index in enumerate(image_index):
					dets = np.asarray(all_boxes[cls_ind][im_ind])
					if dets == []:
						continue
					# the VOCdevkit expects 1-based indices
					for k in range(dets.shape[0]):
						f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
								format(index, dets[k, -1],
									   dets[k, 0] + 1, dets[k, 1] + 1,
									   dets[k, 2] + 1, dets[k, 3] + 1))
import xml.etree.ElementTree as ET
def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
	obj_struct = {}
	obj_struct['name'] = obj.find('name').text
	obj_struct['pose'] = obj.find('pose').text
	obj_struct['truncated'] = int(obj.find('truncated').text)
	obj_struct['difficult'] = int(obj.find('difficult').text)
	bbox = obj.find('bndbox')
	obj_struct['bbox'] = [int(bbox.find('xmin').text),
						  int(bbox.find('ymin').text),
						  int(bbox.find('xmax').text),
						  int(bbox.find('ymax').text)]
	objects.append(obj_struct)

  return objects
def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
	# 11 point metric
	ap = 0.
	for t in np.arange(0., 1.1, 0.1):
	  if np.sum(rec >= t) == 0:
		p = 0
	  else:
		p = np.max(prec[rec >= t])
	  ap = ap + p / 11.
  else:
	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))

	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
	  mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]

	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


# # In[24]:


write_voc_results_file(pascal_classes, all_boxes, image_index)


# # # evaluate on VOC 2010 metric

# # In[25]:


# def python_eval(output_dir='output'):
#         annopath = os.path.join(
#             DEVKIT_PATH,
#             'VOC2007',
#             'Annotations',
#             '{:s}.xml')
#         imagesetfile = os.path.join(
#             DEVKIT_PATH,
#             'VOC2007',
#             'ImageSets',
#             'Main',
#             SET + '.txt')
#         cachedir = os.path.join(DEVKIT_PATH, 'annotations_cache')
#         aps = []
#         # The PASCAL VOC metric changed in 2010.
#         # VOC07 metric is quite old so don't use.
#         use_07_metric = False
#         print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
#         if not os.path.isdir(output_dir):
#             os.mkdir(output_dir)
#         for i, cls in enumerate(PASCAL_CLASSES):
#             if cls == 'bg':
#                 continue
#             filename = get_voc_results_file_template(cls)
#             rec, prec, ap = voc_eval(
#                 filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
#                 use_07_metric=use_07_metric)
#             aps += [ap]
#             print('AP for {} = {:.4f}'.format(cls, ap))
#             with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
#                 pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
#         print('Mean AP = {:.4f}'.format(np.mean(aps)))
#         print('~~~~~~~~')
#         print('Results:')
#         for ap in aps:
#             print('{:.3f}'.format(ap))
#         print('{:.3f}'.format(np.mean(aps)))
#         print('~~~~~~~~')
#         print('')
#         print('--------------------------------------------------------------')
#         print('Results computed with the **unofficial** Python eval code.')
#         print('Results should be very close to the official MATLAB eval code.')
#         print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
#         print('-- Thanks, The Management')
#         print('--------------------------------------------------------------')


# # In[28]:


# # evaluate detections
# python_eval()


# # In[ ]:


# a = '../VOCdevkit/VOC2007/JPEGImages/000585.jpg'
# a[-10:-4]


# # In[ ]:




