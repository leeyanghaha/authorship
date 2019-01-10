import os
import sys
import time

import numpy as np
import tensorflow as tf

from my_method.capsules.models import capsule_model
from my_method.capsules.models import conv_model
from my_method.capsules.input_data.review import review_input_record

FLAGS = tf.flags.FLAGS
os.environ['CUDA_VISIABLE_DEVICES'] = '2'

tf.flags.DEFINE_string('hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'hparams of this experiment.')
tf.flags.DEFINE_integer('max_steps', 1000, 'Number of steps to train.')
tf.flags.DEFINE_string('model', 'capsule',
                       'The model to use for the experiment.'
                       'capsule or baseline')
tf.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus to use.')
tf.flags.DEFINE_string('gpu_names', '[2, ]', 'user gpu ids')
tf.flags.DEFINE_integer('num_trials', 1,
                        'Number of trials for ensemble evaluation.')
tf.flags.DEFINE_integer('save_step', 1500, 'How often to save checkpoints.')
tf.flags.DEFINE_string('summary_dir', '/home/yangl/research/authorship/model/capsule/summary',
                       'Main directory for the experiments.')
tf.flags.DEFINE_bool('train', True, 'Either train the model or test the model.')
tf.flags.DEFINE_bool('validate', False, 'Run trianing/eval in validation mode.')
tf.flags.DEFINE_integer('max_ngram_len', 3500, 'fix input len')


models = {
    'capsule': capsule_model.CapsuleModel,
    'baseline': conv_model.ConvModel,
}


def get_features(split, batch_size):
    input =  review_input_record.input(batch_size=batch_size,
                                              split=split,
                                              max_ngram_len=FLAGS.max_ngram_len)
    return input


def extract_step(path):
    """Returns the step from the file format name of Tensorflow checkpoints.

    Args:
        path: The checkpoint path returned by tf.train.get_checkpoint_state.
        The format is: {ckpnt_name}-{step}

    Returns:
        The last training step number of the checkpoint.
    """
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])


def load_training(saver, session, load_dir):
      """Loads a saved model into current session or initializes the directory.

      If there is no functioning saved model or FLAGS.restart is set, cleans the
      load_dir directory. Otherwise, loads the latest saved checkpoint in load_dir
      to session.

      Args:
        saver: An instance of tf.train.saver to load the model in to the session.
        session: An instance of tf.Session with the built-in model graph.
        load_dir: The directory which is used to load the latest checkpoint.

      Returns:
        The latest saved step.
      """
      if tf.gfile.Exists(load_dir):
          ckpt = tf.train.get_checkpoint_state(load_dir)
          if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)
          else:
            tf.gfile.DeleteRecursively(load_dir)
            tf.gfile.MakeDirs(load_dir)
            prev_step = 0
      else:
          tf.gfile.MakeDirs(load_dir)
          prev_step = 0
      return prev_step


def train_experiment(session, result, writer, last_step, max_steps, saver,
                     summary_dir, save_step):
      """Runs training for up to max_steps and saves the model and summaries.

      Args:
        session: The loaded tf.session with the initialized model.
        result: The resultant operations of the model including train_op.
        writer: The summary writer file.
        last_step: The last trained step.
        max_steps: Maximum number of training iterations.
        saver: An instance of tf.train.saver to save the current model.
        summary_dir: The directory to save the model in it.
        save_step: How often to save the model ckpt.
      """
      step = 0
      for i in range(last_step, max_steps):
        step += 1
        summary, _ = session.run([result.summary, result.train_op])
        writer.add_summary(summary, i)
        if (i + 1) % save_step == 0:
          saver.save(
              session, os.path.join(summary_dir, 'model.ckpt'), global_step=i + 1)


def load_eval(saver, session, load_dir):
  """Loads the latest saved model to the given session.

  Args:
    saver: An instance of tf.train.saver to load the model in to the session.
    session: An instance of tf.Session with the built-in model graph.
    load_dir: The path to the latest checkpoint.

  Returns:
    The latest saved step.
  """
  saver.restore(session, load_dir)
  print('model loaded successfully')
  return extract_step(load_dir)


def eval_experiment(session, result, writer, last_step, max_steps, **kwargs):
  """Evaluates the current model on the test dataset once.

  Evaluates the loaded model on the test data set with batch sizes of 100.
  Aggregates the results and writes one summary point to the summary file.

  Args:
    session: The loaded tf.session with the trained model.
    result: The resultant operations of the model including evaluation metrics.
    writer: The summary writer file.
    last_step: The last trained step.
    max_steps: Maximum number of evaluation iterations.
    **kwargs: Arguments passed by run_experiment but not used in this function.
  """
  del kwargs

  total_correct = 0
  total_almost = 0
  for _ in range(max_steps):
    summary_i, correct, almost = session.run(
        [result.summary, result.correct, result.almost])
    total_correct += correct
    total_almost += almost

  total_false = max_steps * 100 - total_correct
  total_almost_false = max_steps * 100 - total_almost
  summary = tf.Summary.FromString(summary_i)
  summary.value.add(tag='correct_prediction', simple_value=total_correct)
  summary.value.add(tag='wrong_prediction', simple_value=total_false)
  summary.value.add(
      tag='almost_wrong_prediction', simple_value=total_almost_false)
  print('Total wrong predictions: {}, wrong percent: {}%'.format(
      total_false, total_false / max_steps))
  tf.logging.info('Total wrong predictions: {}, wrong percent: {}%'.format(
      total_false, total_false / max_steps))
  writer.add_summary(summary, last_step)


def run_experiment(loader,
                   load_dir,
                   writer,
                   experiment,
                   result,
                   max_steps,
                   save_step=0):
  """Starts a session, loads the model and runs the given experiment on it.

  This is a general wrapper to load a saved model and run an experiment on it.
  An experiment can be a training experiment or an evaluation experiment.
  It starts session, threads and queues and closes them before returning.

  Args:
    loader: A function of prototype (saver, session, load_dir) to load a saved
      checkpoint in load_dir given a session and saver.
    load_dir: The directory to load the previously saved model from it and to
      save the current model in it.
    writer: A tf.summary.FileWriter to add summaries.
    experiment: The function of prototype (session, result, writer, last_step,
      max_steps, saver, load_dir, save_step) which will execute the experiment
      steps from result on the given session.
    result: The resultant final operations of the built model.
    max_steps: Maximum number of experiment iterations.
    save_step: How often the training model should be saved.
  """
  session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  session.run(init_op)

  saver = tf.train.Saver(max_to_keep=1000)
  last_step = loader(saver, session, load_dir)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coord)
  try:
    experiment(
        session=session,
        result=result,
        writer=writer,
        last_step=last_step,
        max_steps=max_steps,
        saver=saver,
        summary_dir=load_dir,
        save_step=save_step)
  except tf.errors.OutOfRangeError:
    tf.logging.info('Finished experiment.')
  finally:
    coord.request_stop()
  coord.join(threads)
  session.close()


def train(hparams, summary_dir, num_gpus, model_type, max_steps, save_step):
  """Trains a model with batch sizes of 128 to FLAGS.max_steps steps.

  It will initialize the model with either previously saved model in the
  summary directory or start from scratch if FLAGS.restart is set or the
  directory is empty.
  The training is distributed on num_gpus GPUs. It writes a summary at every
  step and saves the model every 1500 iterations.

  Args:
    hparams: The hyper parameters to build the model graph.
    summary_dir: The directory to save model and write training summaries.
    num_gpus: Number of GPUs to use for reading data and computation.
    model_type: The model architecture category.
    max_steps: Maximum number of training iterations.
    save_step: How often the training model should be saved.
    data_dir: Directory containing the input data.
    num_targets: Number of objects present in the image.
    dataset: Name of the dataset for the experiments.
    validate: If set, use training-validation set for training.
  """
  summary_dir += '/train/'
  # with tf.Graph().as_default():
    # Build model
  input = get_features(split='train', batch_size=64)
  dataset = input['dataset']
  _num_classes, _ngram_num = input['num_classes'], input['ngram_num']
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()
  model = models[model_type](hparams)


  try:
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        text, label = sess.run(next_element)
            # print(label.shape)

        result, _ = model.multi_gpu(text, label, eval(FLAGS.gpu_names),
                                        _num_classes,
                                        _ngram_num)

            # param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            #     tf.get_default_graph(),
            #     tfprof_options=tf.contrib.tfprof.model_analyzer.
            #         TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            # sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
  except tf.errors.OutOfRangeError:
      pass
  writer = tf.summary.FileWriter(summary_dir)

  run_experiment(load_training, input, summary_dir, writer, train_experiment, result,
                           max_steps, save_step)
  writer.close()





def default_hparams():
  """Builds an HParam object with default hyperparameters."""
  return tf.contrib.training.HParams(
      decay_rate=0.96,
      decay_steps=2000,
      leaky=False,
      learning_rate=0.001,
      loss_type='softmax',
      num_prime_capsules=32,
      padding='SAME',
      remake=False,
      routing=3,
      verbose=False,
  )


def main(_):
  hparams = default_hparams()
  train(hparams, FLAGS.summary_dir, FLAGS.num_gpus, FLAGS.model,
        FLAGS.max_steps, FLAGS.save_step)


if __name__ == '__main__':
    tf.app.run()