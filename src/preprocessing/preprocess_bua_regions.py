# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division

from os import path as os_path
import numpy as np
import json
import cv2
import caffe
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import _get_image_blob, _get_rois_blob
from preprocessor import Preprocessor


# 1) Run image through BottomUp to get region pool5_flat features for each bounding box and the full image
# 2) Do avg pooling over the boxes for each entity so you get a resulting single region
# 3) Concat the bb coords and the number of bbs that made up this region to the featuremap as extra features
# 4) Add the features from 2+3 to this entity_id in a region_dict
# 5) Store the features in a numpy array in the order defined by the entity id list in the raw data
# 6) Save json with region_dict for each IMAGE (these can later be used when packaging the features for each EXAMPLE)
class BottomUpRegionPreprocessor(Preprocessor):
    def __init__(self, image_dir, raw_dir, output_dir, nn_config_path, nn_layout_path, nn_weights_path, device):
        self.image_dir = image_dir
        self.raw_dir = raw_dir
        self.output_dir = output_dir

        # Load the bottom-up nn
        cfg_from_file(nn_config_path)
        self.net = self._load_bottom_up(nn_layout_path, nn_weights_path, device)

    def _preprocess_single(self, image_id):
        # Load image and pass it through the bottom-up net to extract bounding box proposals
        img = cv2.imread(os_path.join(self.image_dir, image_id + '.jpg'))

        # in IMAGE coords: xmin = boxes[i][0], ymin = boxes[i][1], xmax = boxes[i][2], ymax = boxes[i][3]
        boxes, id_to_indices, all_regions, entity_ids = self._load_regions(image_id)

        # Add an extra "box" at the end which covers the full image
        height, width, _ = img.shape
        if len(boxes) > 0:
            boxes = np.concatenate([boxes, np.asarray([[0, 0, width, height]])], axis=0)
        else:
            boxes = np.asarray([[0, 0, width, height]])

        # Extract features from the nn
        all_features = self._extract_features(img, boxes)

        # Combine features that share the same region id
        combined_features = list()
        for idx in entity_ids:
            # Find the indices of the bbs that make up this entity
            bb_indices = tuple([int(x) for x in id_to_indices[idx]])

            # Average over the features for all the bbs that make up this entity
            current_features = all_features[bb_indices, :].mean(axis=0, keepdims=True)
            # Add the relative total min and max x an y for bounding box(es) as well as the number of bbs
            current_bbs = boxes[bb_indices, :]
            spatial_features = np.asarray([[ min(current_bbs[:, 0]) / width, min(current_bbs[:, 1]) / height,
                                             max(current_bbs[:, 2]) / width, max(current_bbs[:, 3]) / height,
                                             len(bb_indices) ]])

            combined_features.append(np.concatenate([current_features, spatial_features], axis=1))

        # Add the full-image box
        combined_features.append(np.concatenate([all_features[(-1,), :],
                                                 np.asarray([[0, 0, 1, 1, 0]], dtype=np.float32)], axis=1))
        combined_features = np.stack(combined_features, axis=0)
        combined_features = combined_features.reshape([len(combined_features), 2048 + 5])

        # Store the data as a single numpy array
        np.save(os_path.join(self.output_dir, image_id), combined_features, allow_pickle=False)

    @staticmethod
    def _load_bottom_up(layout_path, weights_path, device):
        if 'cuda' in device:
            caffe.set_mode_gpu()
            caffe.set_device(0)

        print('Loading bottom-up model...')
        net = caffe.Net(layout_path, caffe.TEST, weights=weights_path)

        if net is not None:
            print('Model loaded.')
        else:
            print('ERROR: Failed to load model:', layout_path, weights_path)

        return net

    # Load all relevant bbs the raw entity data and keep track of which belong to what entity id
    def _load_regions(self, image_id):
        all_entity_regions = list()
        id_to_indices = dict()

        # Read the json data for this image
        with open(os_path.join(self.raw_dir, image_id + '_raw.json'), 'rt') as infile:
            json_data = json.load(infile)
            all_regions = json_data['all_regions']
            entity_ids = json_data['all_entity_ids']

        # Find all the bounding boxes for all regions
        index = 0
        for entity_id in entity_ids:
            for region in all_regions[entity_id]:
                all_entity_regions.append([region['x_min'], region['y_min'], region['x_max'], region['y_max']])
                try:
                    id_to_indices[entity_id].append(index)
                except KeyError:
                    id_to_indices[entity_id] = [index]

                index += 1

        all_entity_regions = np.asarray(all_entity_regions)

        return all_entity_regions, id_to_indices, all_regions, entity_ids

    def _extract_features(self, im, boxes):
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scales = _get_image_blob(im)
        blobs['rois'] = _get_rois_blob(boxes, im_scales)

        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)

        self.net.blobs['data'].reshape(*(blobs['data'].shape))
        if 'im_info' in self.net.blobs:
            self.net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        self.net.blobs['rois'].reshape(*(blobs['rois'].shape))

        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
        if 'im_info' in self.net.blobs:
            forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
        _ = self.net.forward(**forward_kwargs)

        return self.net.blobs['pool5_flat'].data
