# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division

import sys
import os
sys.path.append(os.getcwd())

from speaksee.evaluation import PTBTokenizer, Bleu, Rouge, Meteor, Cider, Spice
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


class TextMetrics:
    def __init__(self):
        self.gts_regions = None
        self.gts_all = None
        self.scorers = None

    def setup(self, metrics, noatt_gt_path, regions_gt_path, example_ids):
        # Load scorers
        self.scorers = dict()
        for metric in metrics:
            if metric == 'BLEU-1':
                self.scorers[metric] = Bleu(n=1)
            if metric == 'BLEU-2':
                self.scorers[metric] = Bleu(n=2)
            if metric == 'BLEU-3':
                self.scorers[metric] = Bleu(n=3)
            if metric == 'BLEU-4':
                self.scorers[metric] = Bleu(n=4)
            if metric == 'ROUGE-L':
                self.scorers[metric] = Rouge()
            if metric == 'METEOR':
                self.scorers[metric] = Meteor()
            if metric == 'CIDEr':
                self.scorers[metric] = Cider()
            if metric == 'SPICE':
                self.scorers[metric] = Spice()

        # Load all ground truth captions for text metrics evaluation
        with open(noatt_gt_path) as gt_file:
            gts_all_base = PTBTokenizer.tokenize(json.load(gt_file)['gts'])

        # Match our example_ids to the base captions (without the _X)
        self.gts_all = dict()
        for example_id in example_ids:
            self.gts_all[example_id] = gts_all_base[example_id.split('_')[0]]

        if regions_gt_path is not None:
            with open(regions_gt_path) as gt_file:
                gts_regions_base = PTBTokenizer.tokenize(json.load(gt_file)['gts'])

            self.gts_regions = dict()
            for example_id in example_ids:
                self.gts_regions[example_id] = gts_regions_base[example_id]

    def prepare_predictions(self, example_ids, predictions):
        generated_captions = dict()
        for i_ex in range(len(predictions)):
            generated_captions[example_ids[i_ex]] = [predictions[i_ex]]

        return generated_captions

    # Calculates for the controllable setting (region_split_scores) and additionally gives the score when NOT taking
    # controllability into account (mixed_captions_score) <- the latter can give some insight into which metrics
    # prefer generic captions over specific ones in the standard Image Captioning setting
    def standard_metrics(self, generated_captions, model_id, split, export_results=False):
        mixed_captions_scores = dict()
        region_split_scores = dict()

        for metric in self.scorers.keys():
            score = self._compute_score(metric, self.gts_all, generated_captions)
            mixed_captions_scores[metric] = score

            if self.gts_regions is not None:
                score = self._compute_score(metric, self.gts_regions, generated_captions)
                region_split_scores[metric] = score

        # Export the results
        if export_results:
            output_path = os.path.join('test_results/', model_id + '_' + split + '.json')
            with open(output_path, 'w') as outfile:
                json.dump({'region_split_scores': region_split_scores, 'mixed_captions_scores': mixed_captions_scores},
                          outfile)

        if self.gts_regions is None:
            region_split_scores = None

        return region_split_scores, mixed_captions_scores

    def _compute_score(self, metric, labels, generated_captions):
        score, details = self.scorers[metric].compute_score(labels, generated_captions)
        if isinstance(score, list):
            # This happens for the BLEU-n scores which return scores from BLEU-1 to BLEU-n
            score = score[-1]

        return score
