# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
from preprocess_bua_regions import BottomUpRegionPreprocessor


_IMG_DIR = os.path.join(os.environ['FLICKR30K_DIR'], 'flickr30k_images')

print("Extracting region features")
burp = BottomUpRegionPreprocessor(image_dir=_IMG_DIR,
                                  raw_dir='../data/bottom-up/features/raw',
                                  output_dir='../data/bottom-up/features/feature_maps',
                                  nn_config_path='../data/bottom-up/models/gt_boxes.yml',
                                  nn_layout_path='../data/bottom-up/models/gt_boxes_features.prototxt',
                                  nn_weights_path='../data/bottom-up/models/vg_flickr30ksplit/resnet101_faster_rcnn_final_iter_380000.caffemodel',
                                  device='cuda:0')
burp.preprocess_all('../data/splits_full.json')
