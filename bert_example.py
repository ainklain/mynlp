
#

import bert
from datetime import datetime
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bert import run_classifier
from bert import optimization
from bert import tokenization

from tensorflow import keras
import os
import re


OUTPUT_DIR = './out/'

def load_directory_data(directory):
    data = {}
    data['sentence'] = []
    data['sentiment'] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), 'r') as f:
            data['sentence'].append(f.read())
            data['sentiment'].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, 'pos'))
    neg_df = load_directory_data(os.path.join(directory, 'neg'))
    pos_df['polarity'] = 1
    neg_df['polarity'] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname='aclImdb.tar.gz',
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset), 'aclImdb', 'train'))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), 'aclImdb', 'test'))

    return train_df, test_df


train, test = download_and_load_datasets()

train = train.sample(5000)
test = train.sample(5000)

DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'polarity'
label_list = [0, 1]

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                             text_a=x[DATA_COLUMN],
                                                                             text_b=None,
                                                                             label=x[LABEL_COLUMN]), axis=1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                             text_a=x[DATA_COLUMN],
                                                                             text_b=None,
                                                                             label=x[LABEL_COLUMN]), axis=1)

# this is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'

def create_tokenizer_from_hub_module():
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature='tokenization_info', as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info['vocab_file'],
                                                  tokenization_info['do_lower_case']])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

MAX_SEQ_LENGTH = 128
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)

    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature='tokens',
        as_dict=True)

    output_layer = bert_outputs['pooled_output']
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        'output_weights', [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable('output_bias', [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope('loss'):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        if is_predicting:
            return (predicted_labels, log_probs)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)

def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps):
    def model_fn(features, labels, mode, params):
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        label_ids = features['label_ids']

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        if not is_predicting:
            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                auc = tf.metrics.auc(label_ids, predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)
                true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
                false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
                false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
                return {'eval_accuracy': accuracy,
                        'f1_score': f1_score,
                        'auc': auc,
                        'precision': precision,
                        'recall': recall,
                        'true_positives': true_pos,
                        'true_negatives': true_neg,
                        'false_positives': false_pos,
                        'false_negatives': false_neg}

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    return model_fn


BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": BATCH_SIZE})

train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

print('Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('Training took time ', datetime.now() - current_time)


test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

estimator.evaluate(input_fn=test_input_fn, steps=None)

def get_prediction(in_sentences):
    labels=['Negative', 'Positive']
    input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in in_sentences]
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

pred_sentences = [
    "That movie was absolutely awful",
    "The acting was a bit lacking",
    "The film was creative and surprising",
    "Absolutely fantastic!"
]

predictions = get_prediction(pred_sentences)

print(predictions)
