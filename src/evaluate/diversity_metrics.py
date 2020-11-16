# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Diversity metrics code adapted from Generating Diverse and Meaningful Captions (Lindh et al., 2018):
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning/blob/master/combined_model/evaluate_model/eval_stats.py

import os
import sqlite3
import json
import numpy as np


_DB_PATH = '../results/diversity_db.sql'
_DATA_DIR = '../data/bottom-up/features/'
_RESULTS_DIR = '../results/'


# Store the ground truth captions in an sqlite table
def store_flickr30k_captions(db_path, gt_captions_file, split):
    caption_data = []
    with open(gt_captions_file) as capfile:
         gts = json.load(capfile)['gts']

    for img_id in gts.keys():
        captions = gts[img_id]
        for i_cap in range(len(captions)):
            caption_data.append((captions[i_cap], img_id + '_' + str(i_cap),))

    table_name = 'gt_' + split
    with sqlite3.connect(db_path) as conn:
        conn.execute('DROP TABLE IF EXISTS ' + table_name)
        conn.execute('CREATE TABLE ' + table_name + ' (caption TEXT, example_id TEXT)')
        conn.executemany('INSERT INTO ' + table_name + ' VALUES (?,?)', caption_data)
        conn.commit()


# Store a model's output from a results file
def store_generated_captions(db_path, captions_file, model_id, split, as_list=True):
    caption_data = []
    with open(captions_file) as capfile:
        generated_captions = json.load(capfile)['generated_captions']

    for example_id in generated_captions.keys():
        if as_list:
            caption_data.append((generated_captions[example_id][0], example_id,))
        else:
            caption_data.append((generated_captions[example_id], example_id,))

    table_name = 'gen_' + split + '_' + model_id
    with sqlite3.connect(db_path) as conn:
        conn.execute('DROP TABLE IF EXISTS ' + table_name)
        conn.execute('CREATE TABLE ' + table_name + ' (caption TEXT, example_id TEXT)')

        conn.executemany('INSERT INTO ' + table_name + ' VALUES (?,?)', caption_data)
        conn.commit()


# Diversity: Calculate Distinct caption stats
def calculate_distinct(db_path, table_name, verbose=True):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(DISTINCT caption) FROM ' + table_name)
        num_distinct = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM ' + table_name)
        num_total = cur.fetchone()[0]
        cur.close()

    fraction_distinct = float(num_distinct) / float(num_total)
    if verbose:
        print("Total generated captions =", num_total)
        print("Number of distinct =", num_distinct)
        print("Fraction distinct =", fraction_distinct)

    return fraction_distinct


# Novel Sentences: percentage of generated captions not seen in the training set.
def calculate_novelty(db_path, table_name_gen, table_name_gts, verbose=True):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*), caption FROM ' + table_name_gen + ' WHERE caption NOT IN (SELECT caption FROM ' + table_name_gts + ')')
        num_novel = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM ' + table_name_gen)
        num_total = cur.fetchone()[0]
        cur.close()

    fraction_novel = float(num_novel)/float(num_total)

    if verbose:
        print("Total generated captions =", num_total)
        print("Number of novel =", num_novel)
        print("Fraction novel =", fraction_novel)
        print("Fraction seen in training data =", 1-fraction_novel)

    return fraction_novel


# Vocabulary Size: number of unique words used in all generated captions
# Returns number of unique words used in the captions of this table
def calculate_vocabulary_usage(db_path, table_name, verbose=True):
    # Build a set of unique words used in the captions of this table+split
    vocab = set()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT caption FROM ' + table_name)
        for caption in cur:
            vocab.update(caption[0].split(' '))
        cur.close()

    if verbose:
        print("Total vocabulary used =", len(vocab))
        if 'UNK' in vocab:
            print('UNK is part of vocab.')
        else:
            print('UNK is NOT part of vocab.')

        # print("Vocab:", vocab)

    return len(vocab)


def calculate_caption_lengths(db_path, table_name, verbose=True):
    caption_lengths = list()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT caption FROM ' + table_name)
        for caption in cur:
            caption_lengths.append(len(caption[0].split(' ')))
        cur.close()

    avg_length = np.asarray(caption_lengths).mean()
    if verbose:
        print("Average caption length = ", avg_length)

    return avg_length


if __name__ == '__main__':
    # Store the ground truth data to compare to
    for split in ['train', 'val', 'test']:
        store_flickr30k_captions(_DB_PATH, os.path.join(_DATA_DIR, 'gt_captions_noatt_' + split + '.json'), split)

    # Calculate ground truth diversity stats
    for split in ['train', 'val', 'test']:
        print('SPLIT:', split)
        calculate_distinct(_DB_PATH, 'gt_' + split)
        calculate_novelty(_DB_PATH, 'gt_' + split, 'gt_train')
        calculate_vocabulary_usage(_DB_PATH, 'gt_' + split)
        calculate_caption_lengths(_DB_PATH, 'gt_' + split)

    # Calculate stats for SCT
    for model in ['sct_ce', 'sct_cider', 'sct_cider_nw']:
        print()
        print('SCT', model)
        table_name = 'gen_test_' + model
        cap_path = os.path.join(_RESULTS_DIR, 'CAPTIONS_' + model + '.json')
        store_generated_captions(_DB_PATH, cap_path, model, 'test')
        calculate_distinct(_DB_PATH, table_name)
        calculate_novelty(_DB_PATH, table_name, 'gt_train')
        calculate_vocabulary_usage(_DB_PATH, table_name)
        calculate_caption_lengths(_DB_PATH, table_name)

    # Calculate our own stats
    for model in ['region_attention', 'average_attention']:
        print()
        print('OURS', model)
        table_name = 'gen_test_' + model
        cap_path = os.path.join(_RESULTS_DIR, 'CAPTIONS_' + model + '.json')
        store_generated_captions(_DB_PATH, cap_path, model, 'test')
        calculate_distinct(_DB_PATH, table_name)
        calculate_novelty(_DB_PATH, table_name, 'gt_train')
        calculate_vocabulary_usage(_DB_PATH, table_name)
        calculate_caption_lengths(_DB_PATH, table_name)
