# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import os
sys.path.append(os.getcwd())

import json
import torch
from torch.utils.data import DataLoader
from caption_generator import CaptionGenerator
from dataloaders import TrainingDataset, CollateTrainingData, InferenceDataset, CollateInferenceData
from evaluate.text_metrics import TextMetrics
from parameter_parsing import parse_parameters

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed. Loss curves will not be logged.")
    tf = None


_CONFIG = None


# Run the entire dataset set once through the training function
def run_epoch(caption_generator, dataloader, update_weights=True):
    epoch_loss_total = 0
    epoch_loss_text = 0
    current_text_logits = None

    caption_generator.set_mode(mode='train')

    batch = None
    num_batches = 0
    for batch in dataloader:
        total_loss, loss_text, current_text_logits = caption_generator.train(
            full_image_features=batch['full_image_features'].to(_CONFIG['device']),
            region_features=batch['region_features'].to(_CONFIG['device']),
            region_start_indices=batch['region_start_indices'].to(_CONFIG['device']),
            captions=batch['captions'].to(_CONFIG['device']),
            update_weights=update_weights)

        epoch_loss_total += total_loss
        epoch_loss_text += loss_text
        num_batches += 1

    epoch_loss_total /= num_batches
    epoch_loss_text /= num_batches

    return epoch_loss_total, epoch_loss_text, batch['example_ids'], current_text_logits


def train(caption_generator, dataloader_train, dataloader_train_inference, dataloader_validation,
          dataloader_validation_inference, num_epochs, loss_writer=None):
    text_metrics_train = TextMetrics()
    text_metrics_train.setup(metrics=_CONFIG['metrics'],
                             regions_gt_path=_CONFIG['regions_gt_path'] + 'train.json' if _CONFIG['model_type'] != 'no_attention' else None,
                             noatt_gt_path=_CONFIG['noatt_gt_path'] + 'train.json',
                             example_ids=dataloader_train_inference.dataset.example_ids)

    text_metrics_val = TextMetrics()
    text_metrics_val.setup(metrics=_CONFIG['metrics'],
                           regions_gt_path=_CONFIG['regions_gt_path'] + 'val.json' if _CONFIG['model_type'] != 'no_attention' else None,
                           noatt_gt_path=_CONFIG['noatt_gt_path'] + 'val.json',
                           example_ids=dataloader_validation_inference.dataset.example_ids)

    best_loss = None
    validation_metrics = _CONFIG['validation_metrics']
    best_metrics = dict()
    for m in validation_metrics:
        best_metrics[m] = None

    for _ in range(num_epochs):
        caption_generator.increment_epoch()

        epoch = caption_generator.get_epoch()
        print("EPOCH", epoch)

        # Run through one epoch of the entire training set and update weights
        loss_total_train, loss_text_train, final_example_ids, final_text_logits = run_epoch(
            caption_generator, dataloader_train, update_weights=True)

        if epoch % _CONFIG['save_freq'] == 0:
            print("Saving latest version of the cg model.")
            caption_generator.save(_CONFIG['checkpoint_path_cg'], 'latest')

        if epoch % _CONFIG['eval_freq'] == 0:
            # Print the final teacher-guided predictions along with their example_ids
            print("Final training example_id and guided prediction output:", final_example_ids[-1],
                  caption_generator.decode_sequences(
                      caption_generator.sample_max(final_text_logits[-1:, :, :])))

            print("loss_total_train", loss_total_train)
            _add_summary_value(loss_writer, 'loss_total_train', loss_total_train, epoch)

            print("loss_text_train", loss_text_train)
            _add_summary_value(loss_writer, 'loss_text_train', loss_text_train, epoch)

            # Calculate the loss on the validation set without updating the weights
            loss_total_val, loss_text_val, final_example_ids, final_text_logits = run_epoch(
                caption_generator, dataloader_validation, update_weights=False)

            # Print the final teacher-guided predictions along with their example_ids
            print("Final validation example_ids and guided prediction output:", final_example_ids[-1],
                  caption_generator.decode_sequences(
                      caption_generator.sample_max(final_text_logits[-1:, :, :])))

            print("loss_total_val", loss_total_val)
            _add_summary_value(loss_writer, 'loss_total_val', loss_total_val, epoch)
            if best_loss is None or best_loss > loss_total_val:
                best_loss = loss_total_val

                print("Saving best CE loss model.")
                caption_generator.save(_CONFIG['checkpoint_path_cg'], 'CE')

            print("loss_text_val", loss_text_val)
            _add_summary_value(loss_writer, 'loss_text_val', loss_text_val, epoch)

            """
            # Uncomment this block for more learning curves on the training set
            # Run in inference mode over the training set
            predictions = inference(caption_generator, dataloader_train_inference)

            # Calculate scores on the standard text metrics from the unguided inference
            generated_captions = text_metrics_train.prepare_predictions(
                example_ids=dataloader_train_inference.dataset.example_ids, predictions=predictions)
            region_split_scores, _ = text_metrics_train.standard_metrics(generated_captions=generated_captions,
                                                                         model_id=_CONFIG['cg_id'],
                                                                         split='train')

            if region_split_scores is not None:
                for metric in region_split_scores:
                    _add_summary_value(loss_writer, metric + '_regions_TRAIN', region_split_scores[metric], epoch)
                    print(metric + '_regions_TRAIN', region_split_scores[metric])

            print("FIRST INFERENCE PREDICTIONS ON TRAINING SET")
            for i_prediction in range(10):
                print(dataloader_train_inference.dataset.example_ids[i_prediction], predictions[i_prediction])

            """

            # Run in inference mode over the validation set
            predictions = inference(caption_generator, dataloader_validation_inference)

            # Calculate scores on the standard text metrics from the unguided inference
            generated_captions = text_metrics_val.prepare_predictions(
                example_ids=dataloader_validation_inference.dataset.example_ids, predictions=predictions)
            region_split_scores, _ = text_metrics_val.standard_metrics(generated_captions=generated_captions,
                                                                       model_id=_CONFIG['cg_id'],
                                                                       split='val')

            if region_split_scores is not None:
                for metric in region_split_scores:
                    _add_summary_value(loss_writer, metric + '_regions_VAL', region_split_scores[metric], epoch)
                    print(metric + '_regions_VAL', region_split_scores[metric])

            print("FIRST INFERENCE PREDICTIONS ON VALIDATION SET")
            for i_prediction in range(10):
                print(dataloader_validation_inference.dataset.example_ids[i_prediction], predictions[i_prediction])

            for metric in validation_metrics:
                if region_split_scores is not None:
                    if best_metrics[metric] is None or best_metrics[metric] < region_split_scores[metric]:
                        best_metrics[metric] = region_split_scores[metric]

                        print("Saving best model for " + metric)
                        caption_generator.save(_CONFIG['checkpoint_path_cg'], metric)


def inference(caption_generator, dataloader):
    caption_generator.set_mode(mode='inference')

    all_predictions = list()

    print("Generating sequences...")
    for batch in dataloader:
        predictions, _, _, _, _ = caption_generator.inference(
            full_image_features=batch['full_image_features'].to(_CONFIG['device']),
            region_features=batch['region_features'].to(_CONFIG['device']),
            region_start_indices=batch['region_start_indices'],
            region_end_indices=batch['region_end_indices'],
            max_seq_length=_CONFIG['max_seq_length'],
            device=_CONFIG['device']
            )

        all_predictions.extend(predictions)

    print("Decoding sequences...")
    all_predictions = caption_generator.decode_sequences(all_predictions)

    return all_predictions


def _add_summary_value(writer, key, value, epoch):
    if writer is not None:
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, epoch)


def launch_training():
    # Create and load the Caption Generator network
    cg = _create_caption_generator()

    print("Creating directory", _CONFIG['checkpoint_path_cg'])
    os.makedirs(_CONFIG['checkpoint_path_cg'], exist_ok=True)
    print("Starting training of CG model", _CONFIG['checkpoint_path_cg'])

    loss_writer = None
    if tf is not None:
        loss_writer = tf.summary.FileWriter(_CONFIG['checkpoint_path_cg'])
    else:
        print("WARNING: Missing TensorFlow, loss and scores will not be logged.")

    # Load the dataset splits
    with open(_CONFIG['splits_path_training'], 'rt') as splits_file:
        example_ids_train = json.load(splits_file)['splits']

    with open(_CONFIG['splits_path_inference'], 'rt') as splits_file:
        example_ids_inference = json.load(splits_file)['splits']

    # Restrict the number of examples in the dataset if requested
    if _CONFIG['limit_train_examples'] > 0:
        if _CONFIG['limit_train_examples'] < len(example_ids_train['train']):
            example_ids_train['train'] = example_ids_train['train'][0:_CONFIG['limit_train_examples']]
        if _CONFIG['limit_train_examples'] < len(example_ids_train['val']):
            example_ids_train['val'] = example_ids_train['val'][0:_CONFIG['limit_train_examples']]
    if _CONFIG['limit_val_examples'] > 0:
        if _CONFIG['limit_val_examples'] < len(example_ids_inference['train']):
            example_ids_inference['train'] = example_ids_inference['train'][0:_CONFIG['limit_val_examples']]
        if _CONFIG['limit_val_examples'] < len(example_ids_inference['val']):
            example_ids_inference['val'] = example_ids_inference['val'][0:_CONFIG['limit_val_examples']]

    # Create the datasets and dataloaders
    dataloader_train = _create_dataloader(example_ids=example_ids_train['train'],
                                          shuffle=True,
                                          is_inference=False,
                                          nexttoken_id=cg.nexttoken_id)
    dataloader_train_inference = _create_dataloader(example_ids=example_ids_inference['train'],
                                                    shuffle=False,
                                                    is_inference=True,
                                                    nexttoken_id=cg.nexttoken_id)

    dataloader_validation = _create_dataloader(example_ids=example_ids_train['val'],
                                               shuffle=False,
                                               is_inference=False,
                                               nexttoken_id=cg.nexttoken_id)
    dataloader_validation_inference = _create_dataloader(example_ids=example_ids_inference['val'],
                                                         shuffle=False,
                                                         is_inference=True,
                                                         nexttoken_id=cg.nexttoken_id)

    try:
        train(cg, dataloader_train, dataloader_train_inference, dataloader_validation, dataloader_validation_inference,
              _CONFIG['num_epochs'], loss_writer)
    except KeyboardInterrupt:
        # Intercept the Ctrl+C command so we can perform some finalization like saving the current model
        print("Training interrupted by user.")

    print("Training finished.")


def launch_test():
    # Create and load the Caption Generator network
    cg = _create_caption_generator()

    print("TEST RESULTS cross-validated on " + _CONFIG['load_type'] + " at epoch " + str(cg.get_epoch()))

    # Load the dataset splits
    with open(_CONFIG['splits_path_inference'], 'rt') as splits_file:
        example_ids = json.load(splits_file)['splits']['test']

    # Restrict the number of examples in the dataset if requested
    if (_CONFIG['limit_test_examples'] > 0) and (_CONFIG['limit_test_examples'] < len(example_ids)):
        example_ids = example_ids[0:_CONFIG['limit_test_examples']]

    dataloader_test = _create_dataloader(example_ids=example_ids,
                                         shuffle=False,
                                         is_inference=True,
                                         nexttoken_id=cg.nexttoken_id)

    # Setup the metrics classes
    text_metrics = TextMetrics()
    text_metrics.setup(metrics=_CONFIG['metrics'],
                       regions_gt_path=_CONFIG['regions_gt_path'] + 'test.json',
                       noatt_gt_path=_CONFIG['noatt_gt_path'] + 'test.json',
                       example_ids=example_ids)

    print("Running inference on test set...")
    predictions = inference(cg, dataloader_test)

    # Calculate scores on the standard text metrics
    print("Calculating text metrics scores...")
    generated_captions = text_metrics.prepare_predictions(example_ids=dataloader_test.dataset.example_ids,
                                                          predictions=predictions)
    region_split_scores, _ = text_metrics.standard_metrics(generated_captions=generated_captions,
                                                           model_id=_CONFIG['cg_id'],
                                                           split='test')

    if region_split_scores is not None:
        for metric in region_split_scores:
            print(metric + '_regions_TEST', region_split_scores[metric])

    # Save all generated captions to the same path the model was loaded from
    with open(_CONFIG['load_path_cg'] + '_CAPTIONS.json', 'wt') as outfile:
        json.dump({'generated_captions': generated_captions}, outfile)


def _create_dataloader(example_ids, shuffle, is_inference, nexttoken_id):
    if is_inference:
        dataset = InferenceDataset(example_ids=example_ids,
                                   data_dir=_CONFIG['data_dir'],
                                   model_type=_CONFIG['model_type'],
                                   spatial_feature_size=_CONFIG['spatial_feature_size'],
                                   drop_num_regions=_CONFIG['drop_num_regions'],
                                   drop_bb_coords=_CONFIG['drop_bb_coords'])
        dataloader = DataLoader(dataset, batch_size=_CONFIG['batch_size'], shuffle=shuffle,
                                num_workers=_CONFIG['num_dataloader_workers'],
                                collate_fn=CollateInferenceData(
                                    region_feature_size=_CONFIG['visual_feature_size']+_CONFIG['spatial_feature_size']))
    else:
        dataset = TrainingDataset(example_ids=example_ids,
                                  data_dir=_CONFIG['data_dir'],
                                  nexttoken_id=nexttoken_id,
                                  model_type=_CONFIG['model_type'],
                                  spatial_feature_size=_CONFIG['spatial_feature_size'],
                                  drop_num_regions=_CONFIG['drop_num_regions'],
                                  drop_bb_coords=_CONFIG['drop_bb_coords'])
        dataloader = DataLoader(dataset, batch_size=_CONFIG['batch_size'], shuffle=shuffle,
                                num_workers=_CONFIG['num_dataloader_workers'],
                                collate_fn=CollateTrainingData(
                                    region_feature_size=_CONFIG['visual_feature_size']+_CONFIG['spatial_feature_size']))

    return dataloader


def _create_caption_generator():
    cg = CaptionGenerator(model_type=_CONFIG['model_type'],
                          vocabulary_path=_CONFIG['vocabulary_path'],
                          word_embedding_size=_CONFIG['word_embedding_size'],
                          visual_feature_size=_CONFIG['visual_feature_size'],
                          spatial_feature_size=_CONFIG['spatial_feature_size'],
                          hidden_size=_CONFIG['cg_hidden_size'],
                          use_all_regions=((_CONFIG['model_type'] == 'region_attention') and
                                           (_CONFIG['use_all_regions'] == 'enforced')),
                          inference_only=(_CONFIG['mode'] == 'test'),
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


def test_next(caption_generator, dataloader):
    nexttoken_id = caption_generator.nexttoken_id
    caption_generator.set_mode(mode='inference')

    tp = 0
    num_actual = 0
    num_predicted = 0
    for batch in dataloader:
        captions = batch['captions'].to(_CONFIG['device'])
        _, _, current_text_logits = caption_generator.train(
            full_image_features=batch['full_image_features'].to(_CONFIG['device']),
            region_features=batch['region_features'].to(_CONFIG['device']),
            region_start_indices=batch['region_start_indices'].to(_CONFIG['device']),
            captions=captions,
            update_weights=False)

        predictions = caption_generator.sample_max(current_text_logits) == nexttoken_id
        predictions *= captions[:, 1:] != 0  # Ignore any predictions in the padded timesteps
        actual = captions[:, 1:] == nexttoken_id

        tp += (predictions * actual).sum()
        num_actual += actual.sum()
        num_predicted += predictions.sum()

    tp = float(tp.cpu().numpy())
    num_predicted = float(num_predicted.cpu().numpy())
    num_actual = float(num_actual.cpu().numpy())

    if num_predicted == 0:
        precision = -1
    else:
        precision = tp / num_predicted

    if num_actual == 0:
        recall = -1
        print("WARNING: No NEXT in the ground truth?")
    else:
        recall = tp / num_actual

    return precision, recall


def launch_test_next():
    # Create and load the Caption Generator network
    cg = _create_caption_generator()

    print("TEST RESULTS cross-validated on " + _CONFIG['load_type'] + " at epoch " + str(cg.get_epoch()))

    # Load the dataset splits
    with open(_CONFIG['splits_path_inference'], 'rt') as splits_file:
        example_ids = json.load(splits_file)['splits']['test']

    # Restrict the number of examples in the dataset if requested
    if (_CONFIG['limit_test_examples'] > 0) and (_CONFIG['limit_test_examples'] < len(example_ids)):
        example_ids = example_ids[0:_CONFIG['limit_test_examples']]

    dataloader_test = _create_dataloader(example_ids=example_ids,
                                         shuffle=False,
                                         is_inference=False,
                                         nexttoken_id=cg.nexttoken_id)

    # Calculate the precision and recall of generating NEXTTOKEN in the teacher-guided setting
    precision, recall = test_next(cg, dataloader_test)

    print("PRECISION", precision)
    print("RECALL", recall)


if __name__ == '__main__':
    _CONFIG = parse_parameters(sys.argv[1:])

    torch.random.manual_seed(_CONFIG['seed'])

    if _CONFIG['mode'] == 'train':
        launch_training()
    elif _CONFIG['mode'] == 'test':
        launch_test()
    elif _CONFIG['mode'] == 'test_next':
        launch_test_next()
