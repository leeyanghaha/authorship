import torch.nn as nn
import utils.function_utils as fu
import collections
import logging
import json

import utils.key_utils as ku
import torch
from torch.utils.data import TensorDataset, DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_reviews():
    file = '/home/leeyang/research/data/CD.json'
    # file = '/home/leeyang/research/data/twitter/training_bert'
    return fu.load_array(file)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, user, product):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.user = user
        self.product = product


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, user, product, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.user = user
        self.product = product
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                user=example.user,
                product=example.product,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(reviews):
    examples = []
    for idx, review in enumerate(reviews):
        line = review[ku.review_text]
        user = review[ku.reviewer_ID]
        product = review[ku.asin]
        line = line.strip()
        examples.append(InputExample(unique_id=idx+1, text_a=line, text_b=None, user=user, product=product))
    return examples


def main():
    output_file = '/home/leeyang/research/model/CD_feature.json'
    bert_model = '/home/leeyang/research/model/bert/bert-base-uncased'
    layers = '-1'
    max_seq_len = 100
    batch_size = 32
    device = torch.device('cuda:0')

    layer_indexes = [int(x) for x in layers.split(",")]
    bert_model_vocab = '/home/leeyang/research/model/bert/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(bert_model_vocab, do_lower_case=True)
    reviews = get_reviews()
    examples = read_examples(reviews)

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_len, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature
    model = BertModel.from_pretrained(bert_model)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)

    model.eval()
    with open(output_file, "w", encoding='utf-8') as writer:
        for input_ids, input_mask, example_indices in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = int(feature.unique_id)
                # feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                output_json['user'] = feature.user
                output_json['product'] = feature.product
                all_out_features = []
                for (i, token) in enumerate(feature.tokens):
                    all_layers = []
                    for (j, layer_index) in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[i]
                        ]
                        all_layers.append(layers)
                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)
                output_json["features"] = all_out_features
                writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
    main()


