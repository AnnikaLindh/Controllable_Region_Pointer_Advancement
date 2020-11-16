# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
from preprocess_entities import EntityRawPreprocessor, VocabularyExtractor, MultiAnnotationSplitExporter
from preprocess_entities import EntityTokenPreprocessor, NextEntityPreprocessor, ArrangeByExamplePreprocessor
from preprocess_entities import create_region_splits, create_gt_dicts

_ENTITIES_DIR = os.path.join(os.environ['FLICKR30K_DIR'], 'Flickr30kEntities')


print("Preprocessing Raw Entities data")
ecp = EntityRawPreprocessor(datadir=_ENTITIES_DIR, output_dir='../data/bottom-up/features/raw')
ecp.preprocess_all('../data/splits_full.json')

print("Extracting Vocabulary")
ve = VocabularyExtractor(split_file='../data/splits_full.json')
ve.generate_vocabulary(input_dir='../data/bottom-up/features/raw', output_dir='../data/bottom-up/features/',
                       minimum_word_occurrences=5)

print("Creating the region splits and mappings...")
create_region_splits(splits_path='../data/splits_full.json', raw_dir='../data/bottom-up/features/raw',
                     outfile='../data/bottom-up/features/region_splits.json')

print("Preprocessing Entity Token tensor data")
etp = EntityTokenPreprocessor(vocabulary_path='../data/bottom-up/features/vocabulary_train.json',
                              input_dir='../data/bottom-up/features/raw/',
                              output_dir='../data/bottom-up/features/token_tensors')
etp.preprocess_all('../data/splits_full.json')

print("Exporting each next-entity index list as a tensor")
nep = NextEntityPreprocessor(input_dir='../data/bottom-up/features/raw',
                             output_dir='../data/bottom-up/features/next_entity_tensors')
nep.preprocess_all('../data/splits_full.json')

print("Generating the splits with each annotation as a separate example")
mase = MultiAnnotationSplitExporter(input_dir='../data/bottom-up/features/token_tensors',
                                    output_path='../data/bottom-up/features/multi_splits_full.json')
mase.export_splits(splits_path='../data/splits_full.json')

print("Arranging all numeric data by example...")
abep = ArrangeByExamplePreprocessor(raw_dir='../data/bottom-up/features/raw/',
                                    feature_dir='../data/bottom-up/features/feature_maps/',
                                    caption_dir='../data/bottom-up/features/token_tensors/',
                                    output_dir='../data/bottom-up/features/numeric_data/')
abep.preprocess_all('../data/splits_full.json')

print("Creating gt caption dicts...")
create_gt_dicts(image_splits_path='../data/splits_full.json',
                region_splits_path='../data/bottom-up/features/region_splits.json',
                raw_dir='../data/bottom-up/features/raw/',
                out_dir='../data/bottom-up/features/')
