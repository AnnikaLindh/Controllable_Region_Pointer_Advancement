# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division

import matplotlib
# Prevent errors due to lack of a connected display
matplotlib.use('Agg')
# ---------------------------------------------------------
import matplotlib.pyplot as plt

import os
from os import path as os_path
import sys
sys.path.append(os.getcwd())
import numpy as np
import json
import csv
from textblob import TextBlob


_SPLITS_PATH = '../data/bottom-up/features/splits_full.json'
_RAW_DIR = '../data/bottom-up/features/raw'
_OUT_DIR = '../data_exploration/chunk_stats'


def _export_hist(data, filepath, title):
    plt.hist(data, bins=100)
    plt.title(title)
    plt.savefig(
        fname=filepath,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()


def _export_bar(tags, tag_numbers, title, filepath, ylim=None):
    # Add the number data from each tag in the correct order
    numbers = list()
    for tag in tags:
        numbers.append(tag_numbers[tag])

    # Create and export the figure
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    if ylim is not None:
        plt.ylim(top=ylim)
    ax.bar(tags, numbers, align='center')
    plt.savefig(
        fname=filepath[:-3] + 'pdf',
        dpi='figure',
        bbox_inches='tight'
    )
    plt.title(title)
    plt.savefig(
        fname=filepath,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()


def chunk_end_stats():
    with open(_SPLITS_PATH, 'rt') as splits_file:
        example_ids = json.load(splits_file)['splits']

    for split in example_ids:
        tag_is_end = dict()
        chunk_lenths = list()
        which_chunk_lengths = {'1': list(), '2': list(), '3': list(), '4+': list()}

        for example_id in example_ids[split]:
            with open(os_path.join(_RAW_DIR, example_id + '_raw.json'), 'rt') as rawfile:
                annotations = json.load(rawfile)['annotations']

            for current_ann in annotations:
                caption = current_ann['tokens'][1:-1]  # Remove BOS and EOS
                next_entity_indices = current_ann['next_entity_indices']

                previous_chunk_end = -1
                chunk_number = 1
                tagged_words = TextBlob(' '.join(caption)).pos_tags

                next_idx_offset = 1  # the indices count from BOS = 0 but we removed BOS before tagging
                tagged_offset = 0  # some words are skipped by the pos tagger

                for i_word in range(len(caption)):
                    current_word, current_tag = tagged_words[i_word+tagged_offset]
                    # Make sure the offset is in sync
                    if caption[i_word] != current_word:
                        tagged_offset -= 1
                        continue

                    is_end_of_chunk = (i_word+next_idx_offset) in next_entity_indices

                    # Increment how often each tag is at the end or not the end of a chunk
                    if current_tag not in tag_is_end:
                        tag_is_end[current_tag] = {True: 0, False: 0}
                    tag_is_end[current_tag][is_end_of_chunk] += 1

                    # Keep track of all the chunk lengths, including where in the chunk order they are
                    if is_end_of_chunk:
                        current_chunk_length = i_word - previous_chunk_end
                        chunk_lenths.append(current_chunk_length)
                        if chunk_number < 4:
                            which_chunk_lengths[str(chunk_number)].append(current_chunk_length)
                        else:
                            which_chunk_lengths['4+'].append(current_chunk_length)

                        previous_chunk_end = i_word
                        chunk_number += 1

        # Print stats and export histograms of chunk length frequencies
        print(split, 'AVG CHUNK LENGTH', np.mean(chunk_lenths))
        output_path = os_path.join(_OUT_DIR, 'hist_chunk_lengths_all_' + split + '.png')
        _export_hist(chunk_lenths, output_path, 'Chunk Length frequencies')
        print(split, 'AVG 1st, 2nd, 3rd, 4+ CHUNK LENGTHS')
        for which_chunk in which_chunk_lengths:
            output_path = os_path.join(_OUT_DIR, 'hist_chunk_lengths_' + which_chunk + '_' + split + '.png')
            _export_hist(which_chunk_lengths[which_chunk], output_path, 'Chunk Length frequencies (chunk ' + which_chunk + ')')
            print(which_chunk, np.mean(which_chunk_lengths[which_chunk]))

        # Prepare the tag stats for csv files and plots
        # (Filtering is based on top tags from initial analysis of the same data)
        csv_rows = list()
        percent_of_tag_is_end = dict()
        top_tags_percent = ['NNS', 'NN', 'NNPS', 'JJS', 'NNP']
        tag_as_end_count = {'other': 0}
        top_tag_counts = ['NN', 'NNS', 'JJ']

        for tag in tag_is_end:
            row = {'tag': tag, 'yes': tag_is_end[tag][True], 'no': tag_is_end[tag][False]}
            csv_rows.append(row)

            if tag in top_tag_counts:
                tag_as_end_count[tag] = row['yes']
            else:
                tag_as_end_count['other'] += row['yes']

            if tag in top_tags_percent:
                percent_of_tag_is_end[tag] = row['yes'] / (row['yes'] + row['no'])

        for tag in top_tag_counts:
            if tag not in tag_as_end_count:
                tag_as_end_count[tag] = 0

        for tag in top_tags_percent:
            if tag not in percent_of_tag_is_end:
                percent_of_tag_is_end[tag] = 0

        # Export bar charts for the end of chunk tag stats
        output_path = os_path.join(_OUT_DIR, 'chart_tag_as_end_count_' + split + '.png')
        _export_bar(tags=top_tag_counts+['other'], tag_numbers=tag_as_end_count,
                    title='Frequency of PoS Tag at the end of a chunk', filepath=output_path)
        output_path = os_path.join(_OUT_DIR, 'chart_percent_of_tag_is_end_' + split + '.png')
        _export_bar(tags=top_tags_percent, tag_numbers=percent_of_tag_is_end,
                    title='PoS Tag positive predictive value for ending a chunk', filepath=output_path, ylim=1.0)

        # Write csv rows for how often each tag was at the end vs not at the end of a chunk
        output_path = os_path.join(_OUT_DIR, 'tag_was_end_' + split + '.csv')
        with open(output_path, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=['tag', 'yes', 'no'])
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)


if __name__ == '__main__':
    chunk_end_stats()
