# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Standalone version of the program where the user can supply their own images and bounding boxes

import sys
import os
sys.path.append(os.getcwd())

import glob
import numpy as np
import cv2
import caffe
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import _get_image_blob, _get_rois_blob
import json
import torch
from caption_generator import CaptionGenerator
from parameter_parsing import parse_parameters


_CONFIG = None


class BottomUp:
    def __init__(self):
        self.net = None

    def load_net(self, layout_path, weights_path, device):
        if device.type == 'cuda':
            caffe.set_mode_gpu()
            caffe.set_device(0)

        print('Loading bottom-up model...')
        net = caffe.Net(layout_path, caffe.TEST, weights=weights_path)

        if net is not None:
            print('Model loaded.')
        else:
            print('ERROR: Failed to load model:', layout_path, weights_path)
            return False

        self.net = net

        return True

    def preprocess_inputs(self, examples_dir):
        batched_full_image_features = list()
        batched_region_features = list()

        os.chdir(examples_dir)
        images = glob.glob('*.jpg')

        for image in images:
            # Load the corresponding region file
            region_path = os.path.join(examples_dir, image[:-3] + 'txt')
            try:
                with open(region_path, 'rt') as region_file:
                    for line in region_file:
                        if len(line) != 0 and not line.startswith('#'):
                            try:
                                regions = json.loads(line)
                                regions = np.asarray(regions)
                                full_image_features, region_features = self._preprocess_single(
                                    image_path=os.path.join(examples_dir, image),
                                    regions=regions)
                                batched_full_image_features.append(full_image_features)
                                batched_region_features.append(region_features)
                            except ValueError:
                                print("WARNING: Malformed line in " + region_path + ": " + line)
                                continue

            except IOError:
                # Skip images without corresponding region files
                print("WARNING: " + image + " has no corresponding " + image[:-3] + "txt file.")
                continue

        if len(batched_full_image_features) == 0:
            print("No valid data files found in dir: " + _CONFIG['examples_dir'])
            return

        batch = dict()
        batch['full_image_features'] = torch.tensor(batched_full_image_features,
                                                    dtype=torch.float32).detach().to(_CONFIG['device'])

        # Add an empty region at the start of the region features list
        batch['region_features'] = [np.zeros([2048+5])]
        # Add the region features from all examples to a single list and keep track of their start and end indices
        batch['region_start_indices'] = list()
        batch['region_end_indices'] = list()
        for feats in batched_region_features:
            if len(feats) > 0:
                batch['region_start_indices'].append(len(batch['region_features']))
                batch['region_features'].extend(feats)
                batch['region_end_indices'].append(len(batch['region_features']))
            else:
                # If this example has no regions, start at index zero where the empty region is
                batch['region_start_indices'].append(0)
                batch['region_end_indices'].append(1)

        # Stack the region features from all examples and convert into a torch tensor
        batch['region_features'] = torch.tensor(np.stack(batch['region_features'], axis=0),
                                                dtype=torch.float32).detach().to(_CONFIG['device'])
        # Convert the index lists into tuples
        batch['region_start_indices'] = tuple(batch['region_start_indices'])
        batch['region_end_indices'] = tuple(batch['region_end_indices'])

        return batch

    def _preprocess_single(self, image_path, regions):
        # Load image and pass it through the bottom-up net to extract bounding box proposals
        img = cv2.imread(image_path)

        # Flatten the list of list of bbs to get a single list of all bbs
        # in IMAGE coords: xmin = boxes[i][0], ymin = boxes[i][1], xmax = boxes[i][2], ymax = boxes[i][3]
        boxes = [bb for region in regions for bb in region]

        # Add an extra "box" at the end which covers the full image
        height, width, _ = img.shape
        if len(boxes) > 0:
            boxes = np.concatenate([boxes, np.asarray([[0, 0, width, height]])], axis=0)
        else:
            boxes = np.asarray([[0, 0, width, height]])

        # Extract features from the nn
        all_img_features = self._extract_features(img, boxes)

        # Full features for each region
        region_features = list()
        i_feature = 0
        for region in regions:
            num_bbs = len(region)
            # If there is more than 1 bb for this region we need to average their img features
            img_features = all_img_features[i_feature:i_feature+num_bbs, :].mean(axis=0, keepdims=True)

            region = np.asarray(region)
            # Get the total min and max for all bbs of this region and divide by total image width and height
            spatial_features = np.asarray([[min(region[:, 0]) / width, min(region[:, 1]) / height,
                                            max(region[:, 2]) / width, max(region[:, 3]) / height,
                                            num_bbs]])

            region_features.append(np.concatenate([img_features, spatial_features], axis=1))

        region_features = np.stack(region_features, axis=0)
        region_features = region_features.reshape([len(region_features), 2048 + 5])

        # Extract the full-image box
        full_image_features = all_img_features[(-1,), :].reshape([1, 2048])

        return full_image_features, region_features

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


def generate_captions(caption_generator, batch):

    print("Generating sequences...")

    predictions, _, _, _, _ = caption_generator.inference(
        full_image_features=batch['full_image_features'].to(_CONFIG['device']),
        region_features=batch['region_features'].to(_CONFIG['device']),
        region_start_indices=batch['region_start_indices'],
        region_end_indices=batch['region_end_indices'],
        max_seq_length=_CONFIG['max_seq_length'],
        device=_CONFIG['device']
        )

    print("Decoding sequences...")
    predictions = caption_generator.decode_sequences(predictions)

    print(predictions)


def _create_caption_generator():
    cg = CaptionGenerator(model_type=_CONFIG['model_type'],
                          vocabulary_path=_CONFIG['vocabulary_path'],
                          word_embedding_size=_CONFIG['word_embedding_size'],
                          visual_feature_size=_CONFIG['visual_feature_size'],
                          spatial_feature_size=_CONFIG['spatial_feature_size'],
                          hidden_size=_CONFIG['cg_hidden_size'],
                          use_all_regions=((_CONFIG['model_type'] == 'region_attention') and
                                           (_CONFIG['use_all_regions'] == 'enforced')),
                          inference_only=True,
                          num_layers=_CONFIG['num_rnn_layers'],
                          learning_rate=_CONFIG['learning_rate'],
                          dropout_lstm=_CONFIG['dropout_lstm'],
                          dropout_word_embedding=_CONFIG['dropout_word_embedding'],
                          l2_weight=_CONFIG['l2_weight'],
                          block_unnecessary_tokens=_CONFIG['block_unnecessary_tokens'],
                          device=_CONFIG['device'])

    if _CONFIG['load_path_cg'] is not None:
        print("Starting from PATH", _CONFIG['load_path_cg'])
        cg.load(checkpoint_path=_CONFIG['load_path_cg'], load_optimizer=_CONFIG['load_cg_optimizer'])

    return cg


if __name__ == '__main__':
    _CONFIG = parse_parameters(sys.argv[1:])

    # Load the Bottom-Up net
    cfg_from_file(_CONFIG['bunet_config_path'])
    bunet = BottomUp()
    bunet.load_net(layout_path=_CONFIG['bunet_layout_path'],
                   weights_path=_CONFIG['bunet_weights_path'],
                   device=_CONFIG['device'])

    batch = bunet.preprocess_inputs(examples_dir=_CONFIG['examples_dir'])

    cg = _create_caption_generator()
    cg.set_mode(mode='inference')

    generate_captions(caption_generator=cg, batch=batch)
