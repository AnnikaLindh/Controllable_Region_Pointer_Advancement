# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Language-Driven Region Pointer Advancement for Controllable Image Captioning
# (Lindh, Ross and Kelleher, 2020)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import json
from operator import itemgetter
import torch
from torch import nn
from torch.autograd.grad_mode import no_grad


class CaptionGenerator:
    def __init__(self, model_type, vocabulary_path, word_embedding_size, visual_feature_size, spatial_feature_size,
                 hidden_size, use_all_regions=False, inference_only=False, num_layers=2,
                 learning_rate=1e-3, dropout_lstm=0.0, dropout_word_embedding=0.0, l2_weight=0.01,
                 block_unnecessary_tokens=False, device='cpu:0'):

        self.model_type = model_type
        self.use_all_regions = use_all_regions
        self.block_unnecessary_tokens = block_unnecessary_tokens

        # Load the vocabulary and get the size
        self.id_to_token = list()
        self.token_to_id = dict()
        self._load_vocabulary(vocabulary_path)

        self.epoch = 0

        # Alias these for easier access
        self.boc_id = self.token_to_id['BOC']
        self.eoc_id = self.token_to_id['EOC']
        self.nexttoken_id = self.token_to_id['NEXTTOKEN']
        self.unk_id = self.token_to_id['UNK']

        # Caption Generation module with an LSTM core
        self.caption_generator = CaptionRNN(vocabulary_size=len(self.id_to_token),
                                            word_embedding_size=word_embedding_size,
                                            visual_feature_size=visual_feature_size,
                                            spatial_feature_size=spatial_feature_size,
                                            hidden_size=hidden_size,
                                            num_layers=num_layers,
                                            dropout_lstm=dropout_lstm,
                                            dropout_word_embedding=dropout_word_embedding).to(device)

        if not inference_only:
            # Setup the optimizer
            self.optimizer = torch.optim.Adam(params=self.caption_generator.parameters(), lr=learning_rate,
                                              weight_decay=l2_weight)
            # CrossEntropyLoss input dims: [batch_size, vocab_size, seq_length]  label dims: [batch_size, seq_length]
            self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean').to(device)

    def _loss_function(self, text_logits=None, text_labels=None):
        loss_text = self.cross_entropy(text_logits, text_labels)

        # Currently there's only one loss type (otherwise add them all to the total loss here)
        total_loss = loss_text

        return total_loss, loss_text

    def set_mode(self, mode):
        if mode == 'train':
            # Set the caption generator to training mode
            self.caption_generator.train()

            # Make sure the gradients are reset before training
            self.optimizer.zero_grad()
        else:
            # Set the caption generator to eval mode
            self.caption_generator.eval()

    def get_epoch(self):
        return self.epoch

    # Samples over logits with dims [batch_size, vocab_size, seq_length]
    # returns token ids with dims [batch_size, seq_length]
    def sample_max(self, logits):
        if self.block_unnecessary_tokens:
            # Prevent the network from predicting BOC, UNK and PADDINGTOKEN by making their values the lowest
            lowest_value = logits.min() - 1.0
            logits[:, [0, self.boc_id, self.unk_id], :] = lowest_value

        sampled_ids = torch.argmax(logits, 1)

        return sampled_ids.detach()

    def train(self, full_image_features, region_features, region_start_indices, captions, update_weights=True):
        batch_size = len(full_image_features)
        region_indices = region_start_indices

        # Initialize the hidden and cell states based on the full image info
        hidden_states, cell_states = self.caption_generator.init_states(full_image_features, batch_size)

        # Generate one step at a time since we need to generate text weights at each step
        text_logits = list()
        for i_step in range(captions.size(1)-1):
            # Get the region features for each example at the current step
            visual_input = region_features[region_indices[i_step], :]

            # Generate the next logits
            output_logits, hidden_states, cell_states, output_embeddings, _ = self.caption_generator(
                captions[:, i_step:(i_step+1)], visual_input, hidden_states, cell_states)

            # Store logits for the loss calculation
            text_logits.append(output_logits)

        # Calculate the loss between the output_logits and a 1-hot of the target tokens
        text_logits = torch.cat(text_logits, dim=2)
        total_loss, loss_text = self._loss_function(text_logits=text_logits, text_labels=captions[:, 1:])

        if update_weights:
            # Backpropagate the total loss for the Caption Generator
            total_loss.backward()

            # Run the optimizer step
            self.optimizer.step()

            # Reset the gradients
            self.optimizer.zero_grad()

        # Return total_loss and the CE loss_text (currently these are the same)
        return total_loss.item(), loss_text.item(), text_logits.detach()

    @torch.no_grad()
    def inference(self, full_image_features, region_features, region_start_indices, region_end_indices,
                  max_seq_length=50, device='cpu:0'):
        batch_size = len(full_image_features)
        region_end_indices = torch.tensor(region_end_indices, dtype=torch.int64, device=device)

        # Keep track of the sampled words, nexttoken positions and text weights
        generated_sequences = torch.zeros(size=[batch_size, max_seq_length], dtype=torch.int64, device=device)
        nexttoken_positions = torch.zeros(size=[batch_size, max_seq_length], dtype=torch.int64, device=device)
        word_text_weights = torch.full(fill_value=-1.0, size=[batch_size, max_seq_length], dtype=torch.float32,
                                       device=device)
        eoc_weights = list()
        nexttoken_weights = list()

        # Keep track of which region each example is currently using
        i_regions = torch.tensor(region_start_indices, dtype=torch.int64, device=device)
        current_region_features = region_features[i_regions, :]

        # Keep track of whether to advance to the next region at the next step for each example in the batch
        advance_attention = torch.zeros([batch_size], dtype=torch.uint8, device=device)
        have_more_regions = (i_regions > 0)  # Zero points to the empty region

        # The first input word will be the start token
        previous_words = torch.tensor([[self.boc_id]] * batch_size, dtype=torch.int64, device=device)

        # Initialize the hidden and cell states based on the full image info
        hidden_states, cell_states = self.caption_generator.init_states(full_image_features, batch_size)

        # Generate words until we either reach max_seq_length or all captions have ended
        unfinished_sequences = torch.ones(size=[batch_size], dtype=torch.uint8, device=device)
        for i_word in range(max_seq_length):
            # If the sum is zero then we will keep the previous visual features
            if advance_attention.sum() > 0:
                # Advance the attention for the chosen examples
                i_regions = i_regions + advance_attention.long()

                # Check if any examples ran out of regions
                out_of_regions_idx = (i_regions >= region_end_indices).nonzero()
                have_more_regions[out_of_regions_idx] = 0
                # Set these examples to permanently use the first region (empty region)
                i_regions[out_of_regions_idx] = 0

                current_region_features = region_features[i_regions, :]

            # Generate the next logits
            output_logits, hidden_states, cell_states, output_embeddings, text_weights = self.caption_generator(
                previous_words, current_region_features, hidden_states, cell_states)

            # Sample the next words
            previous_words = self.sample_max(output_logits)

            # Enforce describing each region by treating EOC as NEXTTOKEN while there are more regions
            if self.use_all_regions:
                ending_early = (previous_words.squeeze(1) == self.eoc_id) * unfinished_sequences * have_more_regions
                if ending_early.sum() > 0:
                    previous_words[ending_early, :] = self.nexttoken_id

            # Find what indices have generated EOC, word or NEXT tokens (for the unfinished sequences)
            eoc_mask = (previous_words.squeeze(1) == self.eoc_id) * unfinished_sequences
            word_mask_indices = ((previous_words.squeeze(1) != self.eoc_id) *
                                 (previous_words.squeeze(1) != self.nexttoken_id) * unfinished_sequences).nonzero()

            # Transfer any word tokens to the generated sequences matrix
            generated_sequences[word_mask_indices, i_word:i_word+1] = previous_words[word_mask_indices, :]

            # Store the current text weight for each actually generated word
            word_text_weights[word_mask_indices, i_word:i_word+1] = text_weights[word_mask_indices, :].squeeze(-1)

            if eoc_mask.sum() > 0:
                # Store text weights for any EOC generated (no need to link these to specific examples)
                eoc_weights.extend([w.item() for w in text_weights[eoc_mask]])

                # Mark sequences as finished
                unfinished_sequences[eoc_mask.nonzero()] = 0

                # Check if all sequences are completed
                if unfinished_sequences.sum() == 0:
                    break

            # Advance to the next region where the output was NEXTTOKEN
            advance_attention = ((previous_words.squeeze(1) == self.nexttoken_id) *
                                 unfinished_sequences * have_more_regions)

            if sum(advance_attention) > 0:
                # Store text weights for any NEXTTOKEN generated (no need to link these to specific examples)
                nexttoken_weights.extend([w.item() for w in text_weights[advance_attention]])

                nexttoken_positions[advance_attention.nonzero(), i_word] = 1

        # Return:
        # * a list of len=batch_size with word token tensors of individual lengths
        # * text weights of actual words
        # * EOC text weights
        # * nexttoken text weights
        # * a list of len=batch_size with np arrays indicating the NEXTTOKEN positions in the generated captions
        return [seq[seq.nonzero()].view(-1) for seq in generated_sequences], \
               [w[w >= 0.0] for w in word_text_weights], eoc_weights, nexttoken_weights, \
               [nextpos.nonzero().flatten().cpu().numpy() for nextpos in nexttoken_positions]

    def increment_epoch(self):
        self.epoch += 1

    # checkpoint_type indicates whether this was the current latest checkpoint or best validation/metric checkpoint, etc
    def save(self, path, checkpoint_type):
        torch.save(self.caption_generator.state_dict(), os_path.join(path, checkpoint_type + '_model.pth'))
        torch.save(self.optimizer.state_dict(), os_path.join(path, checkpoint_type + '_optimizer.pth'))
        with open(os_path.join(path, checkpoint_type + '.txt'), 'at') as log:
            log.write(str(self.epoch) + '\n')

    def load(self, checkpoint_path, load_optimizer):
        self.caption_generator.load_state_dict(torch.load(checkpoint_path + '_model.pth'))
        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(checkpoint_path + '_optimizer.pth'))
        # Load the current epoch for this checkpoint
        with open(checkpoint_path + '.txt', 'rt') as log:
            self.epoch = int(log.read().splitlines()[-1])

    def decode_sequences(self, sequences):
        decoded_sequences = list()

        for seq in sequences:
            if len(seq) == 0:
                decoded_sequences.append('')
            elif len(seq) == 1:
                # When the length is 1, itemgetter does not return a list
                decoded_sequences.append(self.id_to_token[seq[0]])
            else:
                decoded_sequences.append(' '.join(itemgetter(*seq)(self.id_to_token)))

        return decoded_sequences

    def _load_vocabulary(self, vocabulary_path):
        with open(vocabulary_path, 'rt') as vocabulary_file:
            json_data = json.load(vocabulary_file)
            self.id_to_token = json_data['id_to_token']
            self.token_to_id = json_data['token_to_id']


class CaptionRNN(nn.Module):
    """
    CaptionRNN: Recursive caption generating model with the following layers:
        WORD EMBEDDINGS: self.word_embeddings -> embedded_word
        INIT CONTEXT HIDDEN: self.init_hidden_state -> h0
        INIT CONTEXT CELL: self.init_cell_state -> c0
        CONTEXT RNN: self.context_rnn (LSTM) -> hi, ci
        WORD PREDICTION: self.word_prediction (Linear) -> logits over vocabulary
        TEXT WEIGHT: self.text_weighting (Linear) -> sigmoid -> text_weight (image_weight = 1 - text_weight)
    """

    def __init__(self, vocabulary_size, word_embedding_size, visual_feature_size, spatial_feature_size, hidden_size,
                 num_layers=1, dropout_lstm=0.0, dropout_word_embedding=0.0):
        super(CaptionRNN, self).__init__()

        self.num_layers = num_layers
        self.dropout_word_embedding = nn.Dropout(dropout_word_embedding)

        # --- Learned Layers ---

        # WORD EMBEDDINGS
        self.word_embeddings = nn.Embedding(vocabulary_size, word_embedding_size, padding_idx=0)
        # INIT CONTEXT
        self.init_hidden_state = nn.Linear(visual_feature_size, self.num_layers * hidden_size)
        self.init_cell_state = nn.Linear(visual_feature_size, self.num_layers * hidden_size)
        # CONTEXT RNN
        self.context_rnn = nn.LSTM(word_embedding_size + visual_feature_size + spatial_feature_size,
                                   hidden_size, num_layers, dropout=dropout_lstm, batch_first=True)
        # WORD PREDICTION
        self.word_prediction = nn.Linear(hidden_size, vocabulary_size)
        # TEXT/IMAGE WEIGHT
        self.text_weighting = nn.Linear(hidden_size, 1)

        # -----------------------

        self._init_weights()

    def _init_weights(self):
        # WORD EMBEDDINGS
        nn.init.uniform_(self.word_embeddings.weight.data, -1.0, 1.0)

        # INIT CONTEXT
        nn.init.uniform_(self.init_hidden_state.weight.data, -0.1, 0.1)
        nn.init.constant_(self.init_hidden_state.bias.data, 0.0)
        nn.init.uniform_(self.init_cell_state.weight.data, -0.1, 0.1)
        nn.init.constant_(self.init_cell_state.bias.data, 0.0)

        # CONTEXT RNN
        weight_range = 0.1
        for parameter_name, parameter in self.context_rnn.named_parameters():
            if 'weight' in parameter_name:
                nn.init.uniform_(parameter, -weight_range, weight_range)
            elif 'bias' in parameter_name:
                nn.init.constant_(parameter, 0.0)

        # WORD PREDICTION
        nn.init.uniform_(self.word_prediction.weight.data, -0.1, 0.1)
        nn.init.constant_(self.word_prediction.bias.data, 0.0)

        # TEXT/IMAGE WEIGHT
        nn.init.uniform_(self.text_weighting.weight.data, -0.1, 0.1)
        nn.init.constant_(self.text_weighting.bias.data, 0.0)

    def init_states(self, full_image_features, batch_size):
        return self.init_hidden_state(full_image_features).view(self.num_layers, batch_size, -1),\
               self.init_cell_state(full_image_features).view(self.num_layers, batch_size, -1)

    def generate_text_weight(self, hidden_states):
        # text_weighting -> sigmoid = weight placed on previous_words
        return torch.nn.functional.sigmoid(self.text_weighting(hidden_states[-1])).unsqueeze(1)

    def forward(self, previous_words, region_features, hidden_states, cell_states):
        # Get the word embedding for the previous words and apply dropout
        previous_words = self.dropout_word_embedding(self.word_embeddings(previous_words))

        # Generate the weight given to the previous word vs the visual input
        text_weights = self.generate_text_weight(hidden_states)

        # Concatenate the weighted text and opposite-weighted visual input
        combined_input = torch.cat([previous_words * text_weights, region_features.unsqueeze(1) * (1 - text_weights)],
                                   dim=2)

        # Pass the input through one step of the RNN
        output_embeddings, (hidden_states, cell_states) = self.context_rnn(combined_input,
                                                                        (hidden_states, cell_states))

        # Generate the logits for our predictions over the vocabulary
        output_logits = self.word_prediction(output_embeddings.squeeze(1)).unsqueeze(2)

        return output_logits, hidden_states, cell_states, output_embeddings, text_weights
