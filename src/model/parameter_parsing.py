# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import sys
import getopt
import time
import os
import ast


_PARAMETER_DEFAULTS = dict(
    # General settings
    device='cuda:0',
    mode='train',  # 'train' / 'test' / 'test_next'
    model_type='region_attention',  # 'region_attention', 'average_attention', 'no_attention'
    block_unnecessary_tokens=False,  # whether to forbid sampling of BOC, UNK and PADDINGTOKEN
    num_dataloader_workers=2,
    seed=333,

    # Architecture settings
    visual_feature_size=2048,
    spatial_feature_size=5,
    drop_num_regions=False,
    drop_bb_coords=False,
    word_embedding_size=1024,
    num_rnn_layers=2,
    cg_hidden_size=1024,
    limit_train_examples=0,
    limit_val_examples=0,
    limit_test_examples=0,
    max_seq_length=50,

    # Training settings
    num_epochs=300,
    batch_size=100,
    learning_rate=0.00001,
    l2_weight=0.0,
    dropout_lstm=0.7,
    dropout_word_embedding=0.7,

    # Inference settings
    use_all_regions='optional',  # 'optional' / 'enforced', the latter treats EOC as NEXTTOKEN until out of regions

    # Save and load settings
    save_freq=5,  # Save latest checkpoint every save_freq epoch
    eval_freq=10,  # Evaluate every eval_freq epochs and save if validation criteria has improved
    cg_id='caption_generator',  # Today's date and time will automatically be appended
    load_type='CIDEr',  # Which of the save types to load: 'CE' / 'METEOR' / 'CIDEr'
    load_path_cg='',  # '' will translate to None, otherwise load_type will be joined to the path
    load_cg_optimizer=False,

    # Evaluation settings
    metrics="['CIDEr']",  # metrics="['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'METEOR', 'CIDEr', 'SPICE']"
    validation_metrics="['CIDEr']",  # which metrics to save checkpoints for best results on

    # Directories and paths
    checkpoint_path_cg='../cg_checkpoints/',
    splits_path_training='../data/bottom-up/features/multi_splits_full.json',
    splits_path_inference='../data/bottom-up/features/region_splits.json',
    vocabulary_path='../data/bottom-up/features/vocabulary_train.json',
    data_dir='../data/bottom-up/features/numeric_data',
    results_dir='results',
    regions_gt_path='../data/bottom-up/features/gt_captions_unique_',
    noatt_gt_path='../data/bottom-up/features/gt_captions_noatt_',

    # Standalone options
    examples_dir='../standalone_examples',  # Full path to a dir with images and bounding boxes named as x.jpg + x.txt
                      # Each region sequence's bbs should be on a single line as follows:
                      # [[x_min, y_min, x_max, y_max], ...], ...,  [[x_min, y_min, x_max, y_max]]]
                      # Examples:
                      #  2 regions with 2 and 1 boxes: [[[56, 110, 100, 120], [80, 100, 60, 220]], [[10, 4, 30, 100]]]
                      #  2 regions with 1 boxes each: [[[56, 110, 100, 120]], [[80, 100, 60, 220]]]
                      # Empty lines and lines starting with # are ignored
    bunet_config_path='../data/bottom-up/models/gt_boxes.yml',
    bunet_layout_path='../data/bottom-up/models/gt_boxes_features.prototxt',
    bunet_weights_path='../data/bottom-up/models/vg_flickr30ksplit/resnet101_faster_rcnn_final_iter_380000.caffemodel',
)


# Parse commandline options and return as a dictionary with default values for those not specified
def parse_parameters(parameters, verbose=True):
    # Format the parameter key names for use in getopts
    parameter_keys = [param_key + '=' for param_key in _PARAMETER_DEFAULTS.keys()]

    # Convert --option=value to a list of tuples
    parsed_parameters, _ = getopt.getopt(parameters, '', parameter_keys)

    # Remove leading -- from the key names, convert values to their correct types (from string) and store them in a dict
    parsed_parameters = {pair[0][2:]: type(_PARAMETER_DEFAULTS[pair[0][2:]])(pair[1]) for pair in parsed_parameters}

    # Start with the default parameter dict and override any specified parameter values to get our updated dict
    _PARAMETER_DEFAULTS.update(parsed_parameters)

    if verbose:
        print("To replicate, run with the following parameters:")
        print(' '.join(['--' + param_key + '=' + str(_PARAMETER_DEFAULTS[param_key]) for param_key in _PARAMETER_DEFAULTS.keys()]))

    # Fill in derived parameters
    _PARAMETER_DEFAULTS['device'] = torch.device(_PARAMETER_DEFAULTS['device'])
    _PARAMETER_DEFAULTS['cg_id'] += time.strftime('_%Y%m%d_%H%M')  # _YYYYMMDD_HHMM
    _PARAMETER_DEFAULTS['checkpoint_path_cg'] = os.path.join(_PARAMETER_DEFAULTS['checkpoint_path_cg'],
                                                             _PARAMETER_DEFAULTS['cg_id'])

    if _PARAMETER_DEFAULTS['load_path_cg'] == '':
        _PARAMETER_DEFAULTS['load_path_cg'] = None
    else:
        _PARAMETER_DEFAULTS['load_path_cg'] = os.path.join(_PARAMETER_DEFAULTS['load_path_cg'],
                                                           _PARAMETER_DEFAULTS['load_type'])

    _PARAMETER_DEFAULTS['results_dir'] = os.path.join(os.getcwd(), _PARAMETER_DEFAULTS['results_dir'])

    # Convert the metrics strings into lists
    _PARAMETER_DEFAULTS['metrics'] = ast.literal_eval(_PARAMETER_DEFAULTS['metrics'])
    _PARAMETER_DEFAULTS['validation_metrics'] = ast.literal_eval(_PARAMETER_DEFAULTS['validation_metrics'])

    return _PARAMETER_DEFAULTS


if __name__ == '__main__':
    # Run this script directly to test the parameter parsing
    print(parse_parameters(sys.argv[1:]))
