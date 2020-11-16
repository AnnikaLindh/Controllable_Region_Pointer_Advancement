# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division

from xml.etree import ElementTree
from os import path as os_path
import re
import json
import ijson
from pathlib import Path
import glob
import numpy as np
import torch
from textblob import TextBlob
from preprocessor import Preprocessor, _print_missing


# Uncomment these to experiment with merging clothing and bodyparts into their parent regions
_FORBIDDEN_REGION_TYPES = []  # ['clothing', 'bodyparts']
_MERGE_WITH_FORBIDDEN = []  # ['people', 'animals']

_REGPATTERN_REGION_IDS = re.compile('\/EN#(\d+)\/(\S+)')
_REGPATTERN_REPLACE_ENTITIES = re.compile('\[\/en#\d+\/\S+\s(.+?)]')
_REGPATTERN_ALLOWED_CHARACTERS = re.compile('([a-z0-9- ]+|__EOC__)')
_REGPATTERN_EXTRA_WHITESPACE = re.compile(' +')
_REGPATTERN_IMAGE_ID = re.compile('^0*(\d+).')

_RESNET_IMAGE_SIZE = 224


def export_karpathy_splits(infile, outfile):
    split_images = {'train': [], 'val': [], 'test': []}

    with open(infile, 'rt') as f_infile:
        # Loop through all images and assign them to the correct split in the dict
        for image_data in ijson.items(f_infile, 'images.item'):
            split_images[image_data['split']].append(_REGPATTERN_IMAGE_ID.match(image_data['filename']).group(1))

    with open(outfile, 'wt') as f_outfile:
        json.dump({'splits': split_images}, f_outfile)


"""
Figures out which examples have the same regions in the same order and outputs a file with the following info:
1 - splits) Train/val/test splits where only the first of each equivalent example is included.
2 - example_mapping) A dict where the examples in the splits from (1) are keys and the value is list of ints
                     representing their equivalent examples (including the example that is the key)
"""
def create_region_splits(splits_path, raw_dir, outfile):
    region_splits = dict()
    examples_mapping = dict()

    with open(splits_path, 'rt') as splits_file:
        image_splits = json.load(splits_file)['splits']

    for split in image_splits:
        region_splits[split] = list()

        for image_id in image_splits[split]:
            # Get all the region id lists for this image
            with open(os_path.join(raw_dir, image_id + '_raw.json'), 'rt') as rawfile:
                region_id_lists = [ann['region_ids'] for ann in json.load(rawfile)['annotations']]

            # Go through the region id lists and add the examples with unique lists to the split
            for i_region_ids in range(len(region_id_lists)):
                # Find the first occurrence of this exact region id order
                first_idx = region_id_lists.index(region_id_lists[i_region_ids])
                # Construct the example id of the first of the equivalent examples
                first_example = image_id + '_' + str(first_idx)

                # If the first example is the current example, then add it to the new split as a unique example
                if first_idx == i_region_ids:
                    region_splits[split].append(first_example)

                # Add the current idx to the list of examples that are considered copies of the first_example
                try:
                    examples_mapping[first_example].append(i_region_ids)
                except KeyError:
                    examples_mapping[first_example] = [i_region_ids]

    with open(outfile, 'wt') as f_outfile:
        json.dump({'splits': region_splits, 'example_mapping': examples_mapping}, f_outfile)


# Builds one dict of example_id -> [associated captions], and another with image_id -> [all captions]
def create_gt_dicts(image_splits_path, region_splits_path, raw_dir, out_dir):
    with open(image_splits_path, 'rt') as splits_file:
        image_splits = json.load(splits_file)['splits']

    with open(region_splits_path, 'rt') as splits_file:
        example_mapping = json.load(splits_file)['example_mapping']

    for split in image_splits:
        unique_region_gts = dict()
        image_gts = dict()

        for image_id in image_splits[split]:
            # Get all the captions for this image
            with open(os_path.join(raw_dir, image_id + '_raw.json'), 'rt') as rawfile:
                # Skip the BOC and EOC tokens
                captions = [' '.join(ann['tokens'][1:-1]) for ann in json.load(rawfile)['annotations']]

            # Store all captions for no-attention models
            image_gts[image_id] = captions

            # Find the captions beloning to each region-order example
            for i_ex in range(len(captions)):
                example_id = image_id + '_' + str(i_ex)

                try:
                    # Gather all captions with the same region order
                    cap_numbers = example_mapping[example_id]
                    unique_region_gts[example_id] = [captions[i_cap] for i_cap in cap_numbers]
                except KeyError:
                    # The other examples with the same region order are already sorted above
                    pass

        # Store the ground truth captions with only one each of the unique region orders
        with open(os_path.join(out_dir, 'gt_captions_unique_' + split + '.json'), 'wt') as f_outfile:
            json.dump({'gts': unique_region_gts}, f_outfile)

        # Store this split's ground truth captions for models without attention
        with open(os_path.join(out_dir, 'gt_captions_noatt_' + split + '.json'), 'wt') as f_outfile:
            json.dump({'gts': image_gts}, f_outfile)


class EntityRawPreprocessor(Preprocessor):
    def __init__(self, datadir, output_dir):
        self.datadir = datadir
        self.output_dir = output_dir
        self.bb_dir = os_path.join(self.datadir, 'Annotations')
        self.caption_dir = os_path.join(self.datadir, 'Sentences')

    def _preprocess_single(self, image_id):
        # Prepare output json structure
        annotations = []
        all_regions = {}

        with open(os_path.join(self.bb_dir, image_id + '.xml'), 'rb') as region_file:
            region_data = ElementTree.parse(region_file).getroot()

            # Read all regions' info from the xml file
            for region in region_data.findall('object'):
                # Find all bounding boxes for this object (if any)
                bounding_boxes = region.findall('bndbox')

                # If this object has at least 1 bounding box, save this entity
                if len(bounding_boxes) > 0:
                    # There might be multiple bounding boxes for a single entity (such as a group of people) so use list
                    # Use the original image x and y coordinates (needed for the bottom-up ROIs)
                    bounding_boxes = [{'x_min': int(coordinates[0].text),
                                       'y_min': int(coordinates[1].text),
                                       'x_max': int(coordinates[2].text),
                                       'y_max': int(coordinates[3].text)}
                                      for coordinates in bounding_boxes]

                    # Store this bounding box for all entity names
                    entity_names = region.findall('name')
                    for entity_name in entity_names:
                        # If this entity already exists, extend the current bb list with the new
                        if entity_name.text in all_regions:
                            all_regions[entity_name.text].extend(bounding_boxes)
                        else:
                            # If this is a new entity, store the bbs we found in its dict place
                            all_regions[entity_name.text] = bounding_boxes.copy()

        # Clean captions and mark entity locations, and store the entity order
        with open(os_path.join(self.caption_dir, image_id + '.txt')) as caption_file:
            for caption in caption_file.readlines():
                # Keep track of the order that the regions are mentioned in this caption
                current_region_ids = list()

                # Keep track of which entities should be skipped (missing bb, getting merged, or forbidden type)
                skip_entities = list()
                i_chunk = 0

                # Find all entity markers in this caption
                re_matches = re.findall(_REGPATTERN_REGION_IDS, caption)
                previous_type = None
                for (region_id, region_type) in re_matches:
                    if (region_type in _FORBIDDEN_REGION_TYPES) and (previous_type in _MERGE_WITH_FORBIDDEN):
                        # Make sure the previous chunk was NOT already skipped due to lacking a region bb
                        if (i_chunk-1) not in skip_entities:
                            # Skip entity for the previous region's text chunk to merge it with this one
                            skip_entities.append(i_chunk - 1)
                        elif region_id in all_regions:
                            # If the chunk we're merging with had no region bb but this one has, use this region
                            current_region_ids.append(region_id)
                        else:
                            # If there's no bb for this chunk or the one we're merging with we can't end a chunk here
                            skip_entities.append(i_chunk)
                        # We won't update previous_type here since we want to allow chain-merging
                    elif region_id not in all_regions:
                        # Skip entity for the this region's text chunk to merge it with the next one
                        skip_entities.append(i_chunk)
                        # Update previous_type so we won't merge with this region's text chunk by mistake
                        previous_type = region_type
                    else:
                        # Keep this region
                        current_region_ids.append(region_id)

                        # Update previous_type when we actually add a regions
                        previous_type = region_type

                    i_chunk += 1

                # Lowercase the caption
                caption = caption.lower()

                # Replace entity markers with the entity text and the nextentity marker
                caption = re.sub(_REGPATTERN_REPLACE_ENTITIES, r'\1 nextentity', caption)

                # Keep only allowed characters
                caption = re.findall(_REGPATTERN_ALLOWED_CHARACTERS, caption)
                caption = ''.join(caption)

                # Remove extra (double) whitespaces, and any at the start and end
                caption = re.sub(_REGPATTERN_EXTRA_WHITESPACE, ' ', caption).strip()

                # Build the list of indices for the nextentity markers while removing them from the text
                next_entity_indices = list()
                cleaned_caption = list()
                i_cleaned_words = 0  # The first token will be BOC which is at index=0
                i_markers = 0
                for current_word in TextBlob(caption).words:
                    if current_word == "nextentity":
                        # Skip markers for chunks that will be merged or that have no bb data
                        if i_markers not in skip_entities:
                            # Mark the last real word's index as where to request to a new entity attention map
                            next_entity_indices.append(i_cleaned_words)

                        # Increment the nextentity marker index
                        i_markers += 1
                    else:
                        # Add normal words to the cleaned caption and increment the word index
                        cleaned_caption.append(current_word)
                        i_cleaned_words += 1

                # Add Beginning Of Caption and End Of Caption tokens
                cleaned_caption = ['BOC'] + cleaned_caption + ['EOC']

                # Add full caption, tokenized caption, next entity indices and regions to the annotation structure
                annotations.append({'caption': ' '.join(cleaned_caption), 'tokens': cleaned_caption,
                                    'next_entity_indices': next_entity_indices, 'region_ids': current_region_ids})

        with open(os_path.join(self.output_dir, image_id + '_raw.json'), 'wt') as outfile:
            json.dump({'all_regions': all_regions, 'all_entity_ids': list(all_regions.keys()), 'annotations': annotations}, outfile)


class VocabularyExtractor:
    def __init__(self, split_file):
        self.vocabulary = None

        # Load the dataset splits
        with open(split_file, 'rt') as f_split_file:
            self.splits = json.load(f_split_file)['splits']

    def generate_vocabulary(self, input_dir, output_dir, minimum_word_occurrences):
        # Use these special tokens
        special_tokens = ['UNK', 'BOC', 'EOC', 'NEXTTOKEN']

        # Keep track of image files that were not found for each split
        not_found = {}
        num_not_found = 0

        # Loop through each dataset split and all the image ids belonging to it
        for current_split in self.splits.keys():
            # Reset the vocabulary and not-found list for each split
            self.vocabulary = {}
            current_not_found = []

            # Extract the token counts from all images in this split
            image_ids = self.splits[current_split]
            for image_id in image_ids:
                try:
                    self._extract_tokens(os_path.join(input_dir, image_id + '_raw.json'))
                except FileNotFoundError:
                    current_not_found.append(image_id)

            # Store a list of image_ids without a file
            not_found[current_split] = current_not_found
            num_not_found += len(current_not_found)

            # Store the full token list with word counts before pruning
            with open(os_path.join(output_dir, 'word_counts_' + current_split + '.json'), 'wt') as outfile:
                json.dump({'word_counts': self.vocabulary}, outfile)

            # Initialize the vocabulary and add the padding token at the first position
            id_to_token = []
            token_to_id = dict()
            token_to_id['PADDINGTOKEN'] = 0
            id_to_token.append('PADDINGTOKEN')

            # Assign ids to all tokens with a minimum number of occurrences
            i_token = 1
            for token in self.vocabulary.keys():
                if self.vocabulary[token] >= minimum_word_occurrences and token not in special_tokens:
                    # Keep this token and give it an id
                    token_to_id[token] = i_token

                    # Add the token to the list for reverse lookup
                    id_to_token.append(token)

                    # Increment for the next id
                    i_token += 1

            # Add any special tokens at the end so they can be easily dropped if desired
            for token in special_tokens:
                # Make sure we haven't named our special tokens to something already in the normal caption text
                assert token not in token_to_id, "WARNING: Special token {} appears in captions.".format(token)

                token_to_id[token] = i_token

                # Add the token to the list for reverse lookup
                id_to_token.append(token)

                # Increment for the next id
                i_token += 1

            # Store the vocabulary both as a list with id to token mapping and a dict with token to id mapping
            with open(os_path.join(output_dir, 'vocabulary_' + current_split + '.json'), 'wt') as outfile:
                json.dump({'id_to_token': id_to_token, 'token_to_id': token_to_id}, outfile)

        _print_missing(not_found, num_not_found)

    def _extract_tokens(self, image_file):
        # Read the json data for this image
        with open(image_file, 'rt') as infile:
            json_data = json.load(infile)

        # Go through all tokens in all annotations and increment their counts in the vocabulary
        for annotation in json_data['annotations']:
            for token in annotation['tokens']:
                self._increment_token(token)

    def _increment_token(self, token):
        if token in self.vocabulary:
            # Increment the count of how many times this token was found
            self.vocabulary[token] += 1
        else:
            # Add the new token and give it a count of one
            self.vocabulary[token] = 1


class NextEntityPreprocessor(Preprocessor):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def _preprocess_single(self, image_id):
        # Read the json data for this image
        json_data = None
        print("A", os_path.join(self.input_dir, image_id + '_raw.json'))
        with open(os_path.join(self.input_dir, image_id + '_raw.json'), 'rt') as infile:
            json_data = json.load(infile)

        # Extract the next-entities parts and export them as tensors in individual files per annotation
        for i_annotation in range(len(json_data['annotations'])):
            # Convert the list into a tensor
            next_entity_indicies = torch.tensor(json_data['annotations'][i_annotation]['next_entity_indices'],
                                                dtype=torch.int64)

            print("B", os_path.join(self.output_dir, image_id + '_' + str(i_annotation) + '.pt'))
            print('tensor data', next_entity_indicies)
            # Save each annotation's tensor to its own file to treat them as individual examples
            torch.save(next_entity_indicies, os_path.join(self.output_dir, image_id + '_' + str(i_annotation) + '.pt'))


class EntityTokenPreprocessor(Preprocessor):
    def __init__(self, vocabulary_path, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Load the vocabulary
        with open(vocabulary_path, 'rt') as vocabulary_file:
            self.vocabulary = json.load(vocabulary_file)['token_to_id']

    def _preprocess_single(self, image_id):
        # Read the json data for this image
        json_data = None
        with open(os_path.join(self.input_dir, image_id + '_raw.json'), 'rt') as infile:
            json_data = json.load(infile)

        # Convert tokens into ids
        for i_annotation in range(len(json_data['annotations'])):
            # Convert all tokens into ids
            token_ids = list()
            for token in json_data['annotations'][i_annotation]['tokens']:
                # Get the id of the next token (or UNK) and add to the list
                try:
                    current_id = self.vocabulary[token]
                except KeyError:
                    # Replace out-of-vocabulary tokens with UNK
                    current_id = self.vocabulary['UNK']

                # Add the current id to the preprocessed caption
                token_ids.append(current_id)

            # Convert the token id list into a LongTensor
            token_ids = torch.tensor(token_ids, dtype=torch.int64)

            # Save each annotation's token id list to its own file to treat them as individual examples
            torch.save(token_ids, os_path.join(self.output_dir, image_id + '_' + str(i_annotation) + '.pt'))


class MultiAnnotationSplitExporter:
    def __init__(self, input_dir, output_path):
        self.input_dir = input_dir
        self.output_path = output_path

    def export_splits(self, splits_path):
        multi_splits = {'train': list(), 'val': list(), 'test': list()}

        with open(splits_path, 'rt') as splits_file:
            single_splits = json.load(splits_file)['splits']

        # Get all the id_<annotation_number> versions for each image_id to treat each annotation as a separate example
        for current_split in single_splits.keys():
            for image_id in single_splits[current_split]:
                multi_splits[current_split].extend([Path(x).stem for x in
                                                    glob.glob(os_path.join(self.input_dir, image_id + "_*"))])

        # Export the new splits file with all annotations as separate examples
        with open(self.output_path, 'wt') as outfile:
            json.dump({'splits': multi_splits}, outfile)


# Store all the number data for a single EXAMPLE together
class ArrangeByExamplePreprocessor(Preprocessor):
    def __init__(self, raw_dir, feature_dir, caption_dir, output_dir):
        self.raw_dir = raw_dir
        self.feature_dir = feature_dir
        self.caption_dir = caption_dir
        self.output_dir = output_dir

    def _preprocess_single(self, image_id):
        # Read the json data for this image
        with open(os_path.join(self.raw_dir, image_id + '_raw.json'), 'rt') as infile:
            json_data = json.load(infile)
            entity_ids = json_data['all_entity_ids']
            annotations = json_data['annotations']

        # Build dict to lookup the feature index from the entity id
        entity_id_to_index = dict()
        for i_entity_id in range(len(entity_ids)):
            entity_id_to_index[entity_ids[i_entity_id]] = i_entity_id

        # Load the region features
        all_features = np.load(os_path.join(self.feature_dir, image_id + '.npy'))

        # Store only the relevant features for each example, in the order that they appear in the caption
        i_example = 0
        for ann in annotations:
            region_indices = tuple([entity_id_to_index[entity_id] for entity_id in ann['region_ids']])
            region_features = all_features[region_indices, :]

            caption_tokens = torch.load(os_path.join(self.caption_dir, image_id + '_' + str(i_example) + '.pt')).numpy()

            # Store the arranged features PER EXAMPLE
            np.savez(os_path.join(self.output_dir, image_id + '_' + str(i_example)),
                     full_image_features=all_features[-1, :-5],
                     region_features=region_features,
                     caption_tokens=caption_tokens,
                     next_entity_indices=np.asarray(ann['next_entity_indices']))

            i_example += 1
