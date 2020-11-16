# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TrainingDataset(Dataset):
    """
    Loads all the relevant data from disk and prepares it for collating.
    Only the list of all example ids is permanently stored in memory.

    Arguments:
        example_ids (list): full list of examples in the form of image ids with the annotation number at the end
        data_dir (string): directory where the data files are located
        nexttoken_id (int): the id number of the NEXTTOKEN in the vocabulary
        model_type (string): region_attention (full model) or average_attention (ablation model)
        spatial_feature_size (int): number of features at the end after the the bottom-up visual features
        drop_num_regions (bool): set to false to disregard the feature saying how many sub-regions make up this region
        drop_bb_coords (string): set to false to disregard the min and max x,y of this region
    """

    def __init__(self, example_ids, data_dir, nexttoken_id, model_type='region_attention', spatial_feature_size=5,
                 drop_num_regions=False, drop_bb_coords=False):
        self.example_ids = example_ids
        self.data_dir = data_dir
        self.nexttoken_id = nexttoken_id
        self.model_type = model_type
        self.spatial_feature_size = spatial_feature_size
        self.drop_num_regions = drop_num_regions
        self.drop_bb_coords = drop_bb_coords

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        return self.load(self.example_ids[index])

    def get_example_id(self, index):
        return self.example_ids[index]

    def load(self, example_id):
        data = np.load(os_path.join(self.data_dir, example_id + '.npz'))

        # Add the NEXT token at the end of each chunk
        if len(data['next_entity_indices']) > 0:
            next_entity_indices = data['next_entity_indices']
            text_with_next = list()
            for i_word in range(len(data['caption_tokens'])):
                text_with_next.append(data['caption_tokens'][i_word].item())
                if i_word in next_entity_indices:
                    text_with_next.append(self.nexttoken_id)
            caption = torch.tensor(text_with_next)

            # Update the location of the NEXT tokens after the shift with each injected NEXT token
            next_entity_indices = (caption == self.nexttoken_id).nonzero().view(-1).numpy()
        else:
            caption = torch.tensor(data['caption_tokens'])
            next_entity_indices = data['next_entity_indices']

        region_features = data['region_features']
        num_regions = len(region_features)
        if num_regions > 0:
            if self.model_type == 'no_attention':
                # Fill all regions with the full image features, padding with zeros to the region feature size
                full_image_padded = np.pad(np.expand_dims(data['full_image_features'], axis=0),
                                           pad_width=((0, 0), (0, self.spatial_feature_size)), mode='constant',
                                           constant_values=0)
                region_features = np.repeat(full_image_padded, num_regions, axis=0)
            elif self.model_type == 'average_attention':
                # Fill all regions with the average visual features, padding with zeros to the region feature size
                avg_regions = region_features.mean(axis=0, keepdims=True)[:, :-5]
                avg_regions_padded = np.pad(avg_regions,
                                            pad_width=((0, 0), (0, self.spatial_feature_size)), mode='constant',
                                            constant_values=0)
                region_features = np.repeat(avg_regions_padded, num_regions, axis=0)
            else:
                if self.drop_num_regions:
                    if self.drop_bb_coords:
                        # Drop all the spatial features
                        region_features = region_features[:, :-5]
                    else:
                        # Drop the final spatial features (number of regions that were combined into a single region)
                        region_features = region_features[:, :-1]
                elif self.drop_bb_coords:
                    # Drop the first 4 spatial features (relative coords) by combining two slice-ranges
                    total_size = len(region_features[0])
                    region_features = region_features[:, np.r_[0:total_size-5, total_size-1:total_size]]

        return {'example_id': example_id, 'full_image_features': data['full_image_features'],
                'region_features': region_features, 'caption': caption,
                'next_entity_indices': next_entity_indices}


class CollateTrainingData:
    """
    Takes a batch of data example dicts and return a dict where each entry is batched.
    The collated batch contains the following entries:
        example_ids          - list of example ids in this batch [batch_size]
        captions             - padded torch matrix with caption token ids [batch_size, max_seq_length]
        full_image_features  - torch matrix of full image features [batch_size, 2048]
        region_features      - torch matrix of region features (incl spatial features) [1 + num_all_regions, 2048 + 5]
        region_start_indices - torch matrix of current region index for each example at each timestep [max_input_length, batch_size]
    """
    def __init__(self, region_feature_size):
        self.region_feature_size = region_feature_size

    def __call__(self, batch):
        batch_size = len(batch)

        collated_batch = dict()

        # -- EXAMPLE_ID
        # Turn into a single list
        collated_batch['example_ids'] = [batch[i]['example_id'] for i in range(batch_size)]

        # -- CAPTIONS
        # Pad the tokens to equal lengths and stack them
        collated_batch['captions'] = pad_sequence([batch[i]['caption'] for i in range(batch_size)],
                                                  batch_first=True).detach()

        # -- FULL_IMAGE_FEATURES
        # Gather all and convert into a single torch matrix
        collated_batch['full_image_features'] = torch.tensor(
            [batch[i]['full_image_features'] for i in range(batch_size)], dtype=torch.float32).detach()

        # -- REGION_FEATURES
        # Add an empty region at the start of the region features list
        collated_batch['region_features'] = [np.zeros([self.region_feature_size])]
        region_start_indices = list()
        # Add the region features from all examples to a single list but keep track of where each example starts
        for i_batch in range(batch_size):
            feats = batch[i_batch]['region_features']
            if len(feats) > 0:
                region_start_indices.append(len(collated_batch['region_features']))
                collated_batch['region_features'].extend(feats)
            else:
                # If this example has no regions, start at index zero where the empty region is
                region_start_indices.append(0)

        # Stack the region features from all examples and convert into a torch tensor
        collated_batch['region_features'] = torch.tensor(np.stack(collated_batch['region_features'], axis=0),
                                                         dtype=torch.float32).detach()

        # -- REGION_INDICES
        # Build a matrix [max_input_length, batch_size] to store the region indices used at each text step
        max_input_length = collated_batch['captions'].size(1) - 1  # -1 because the final token is not part of the input
        collated_batch['region_start_indices'] = list()
        for i_batch in range(batch_size):
            current_indicies = list()
            previous_idx = 0
            current_region_index = region_start_indices[i_batch]
            # indicies [2, 5, 7] will give [1,1, 2,2,2, 3,3]
            for idx in batch[i_batch]['next_entity_indices']:
                current_indicies.extend([current_region_index] * (idx - previous_idx))
                previous_idx = idx
                current_region_index += 1

            # Use the empty region for any steps beyond the final entity
            current_indicies.extend([0] * (max_input_length - len(current_indicies)))

            collated_batch['region_start_indices'].append(current_indicies)

        # Turn list of lists into torch tensor and transpose into [max_input_length, batch_size]
        collated_batch['region_start_indices'] = torch.tensor(collated_batch['region_start_indices']).transpose(1, 0).detach()

        return collated_batch


class InferenceDataset(Dataset):
    """
    Loads all the relevant data from disk and prepares it for collating.
    Only the list of all example ids is permanently stored in memory.

    Arguments:
        example_ids (list): full list of examples in the form of image ids with the annotation number at the end
        data_dir (string): directory where the data files are located
        model_type (string): region_attention (full model) or average_attention (ablation model)
        spatial_feature_size (int): number of features at the end after the the bottom-up visual features
        drop_num_regions (bool): set to false to disregard the feature saying how many sub-regions make up this region
        drop_bb_coords (string): set to false to disregard the min and max x,y of this region
    """

    def __init__(self, example_ids, data_dir, model_type='region_attention', spatial_feature_size=5,
                 drop_num_regions=False, drop_bb_coords=False):
        self.example_ids = example_ids
        self.data_dir = data_dir
        self.model_type = model_type
        self.spatial_feature_size = spatial_feature_size
        self.drop_num_regions = drop_num_regions
        self.drop_bb_coords = drop_bb_coords

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        return self.load(self.example_ids[index])

    def get_example_id(self, index):
        return self.example_ids[index]

    def load(self, example_id):
        data = np.load(os_path.join(self.data_dir, example_id + '.npz'))

        region_features = data['region_features']
        num_regions = len(region_features)
        if num_regions > 0:
            if self.model_type == 'no_attention':
                # Fill all regions with the full image features, padding with zeros to the region feature size
                full_image_padded = np.pad(np.expand_dims(data['full_image_features'], axis=0),
                                           pad_width=((0, 0), (0, self.spatial_feature_size)), mode='constant',
                                           constant_values=0)
                region_features = np.repeat(full_image_padded, num_regions, axis=0)
            elif self.model_type == 'average_attention':
                # Fill all regions with the average visual features, padding with zeros to the region feature size
                avg_regions = region_features.mean(axis=0, keepdims=True)[:, :-5]
                avg_regions_padded = np.pad(avg_regions,
                                            pad_width=((0, 0), (0, self.spatial_feature_size)), mode='constant',
                                            constant_values=0)
                region_features = np.repeat(avg_regions_padded, num_regions, axis=0)
            else:
                if self.drop_num_regions:
                    if self.drop_bb_coords:
                        # Drop all the spatial features
                        region_features = region_features[:, :-5]
                    else:
                        # Drop the final spatial features (number of regions that were combined into a single region)
                        region_features = region_features[:, :-1]
                elif self.drop_bb_coords:
                    # Drop the first 4 spatial features (relative coords) by combining two slice-ranges
                    total_size = len(region_features[0])
                    region_features = region_features[:, np.r_[0:total_size-5, total_size-1:total_size]]

        return {'example_id': example_id, 'full_image_features': data['full_image_features'],
                'region_features': region_features}


class CollateInferenceData:
    """
    Takes a batch of data example dicts and return a dict where each entry is batched.
    The collated batch contains the following entries:
        example_ids          - list of example ids in this batch [batch_size]
        full_image_features  - torch matrix of full image features [batch_size, 2048]
        region_features      - torch matrix of region features (incl spatial features) [1 + num_all_regions, 2048 + 5]
        region_start_indices - tuple containing the indices of the first region features for each example [batch_size]
        region_end_indices   - tuple containing the indices of 1 beyond the last features for each example [batch_size]
    """
    def __init__(self, region_feature_size):
        self.region_feature_size = region_feature_size

    def __call__(self, batch):
        batch_size = len(batch)

        collated_batch = dict()

        # -- EXAMPLE_ID
        # Turn into a single list
        collated_batch['example_ids'] = [batch[i]['example_id'] for i in range(batch_size)]

        # -- FULL_IMAGE_FEATURES
        # Gather all and convert into a single torch matrix
        collated_batch['full_image_features'] = torch.tensor(
            [batch[i]['full_image_features'] for i in range(batch_size)], dtype=torch.float32).detach()

        # -- REGION_FEATURES -- REGION_START_INDICES -- REGION_END_INDICES
        # Add an empty region at the start of the region features list
        collated_batch['region_features'] = [np.zeros([self.region_feature_size])]
        # Add the region features from all examples to a single list and keep track of their start and end indices
        collated_batch['region_start_indices'] = list()
        collated_batch['region_end_indices'] = list()
        for i_batch in range(batch_size):
            feats = batch[i_batch]['region_features']
            if len(feats) > 0:
                collated_batch['region_start_indices'].append(len(collated_batch['region_features']))
                collated_batch['region_features'].extend(feats)
                collated_batch['region_end_indices'].append(len(collated_batch['region_features']))
            else:
                # If this example has no regions, start at index zero where the empty region is
                collated_batch['region_start_indices'].append(0)
                collated_batch['region_end_indices'].append(1)

        # Stack the region features from all examples and convert into a torch tensor
        collated_batch['region_features'] = torch.tensor(np.stack(collated_batch['region_features'], axis=0),
                                                         dtype=torch.float32).detach()
        # Convert the index lists into tuples
        collated_batch['region_start_indices'] = tuple(collated_batch['region_start_indices'])
        collated_batch['region_end_indices'] = tuple(collated_batch['region_end_indices'])

        return collated_batch
