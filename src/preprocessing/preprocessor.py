# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import json


# Python 2 does not have FileNotFoundError so this is to make it compatible
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def _print_missing(not_found, num_not_found):
    if num_not_found > 20:
        # If a lot of files are missing, show a summary
        for split in not_found.keys():
            print(len(not_found[split]), " data files missing for split ", split)
    elif num_not_found > 0:
        print("The following image_ids did not have a matching data file:")
        print(not_found)


# The base class for all Preprocessors with a general preprocess_all function
# Descendants should override the function _preprocess_single(self, image_id) and optionally _post_processing(self)
class Preprocessor:
    def preprocess_all(self, splits_path):
        # Load the dataset splits
        splits = None
        with open(splits_path, 'rt') as splits_file:
            splits = json.load(splits_file)['splits']

        # Keep track of which images don't have preprocessed files
        not_found = {}
        num_not_found = 0

        for current_split in splits.keys():
            current_not_found = []
            for image_id in splits[current_split]:
                try:
                    self._preprocess_single(image_id)
                except FileNotFoundError:
                    current_not_found.append(image_id)

            # Store a list of image_ids without a file
            not_found[current_split] = current_not_found
            num_not_found += len(current_not_found)

        _print_missing(not_found, num_not_found)

        self._post_processing()

    # This function must be overridden with the actual implementation of the preprocessing done per file
    def _preprocess_single(self, image_id):
        assert False, "Missing implementation for _preprocess_single(self, image_id)"

    # Override this function to perform some final action after processing all files
    def _post_processing(self):
        pass
