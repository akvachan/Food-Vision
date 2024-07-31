import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import datetime
import functools
import json
import os
import time
from typing import Tuple


# Time helper function
def timer(func):
    """
    Print the runtime of the decorated function.
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.4f} secs")
        return value
    return wrapper_timer


def getvar(var: str, vars_file=os.path.join(".", ".credentials")) -> str:
    """
    Credentials, secrets fetching function.
    """
    with open(vars_file) as f:
        d = json.load(f)
        return d[var]


@timer
def load(dataset_name: str, dataset_path: str) -> \
        Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """
    Loads the specified dataset from TensorFlow Datasets.
    """
    (test_set, val_set, train_set), ds_info = tfds.load(
      name=dataset_name,
      data_dir=dataset_path,
      split=["validation", "train[0%:15%]", "train[15%:]"],
      shuffle_files=True,
      as_supervised=True,
      with_info=True
    )
    return test_set, val_set, train_set, ds_info


def random_sample(data: tf.data.Dataset, ds_info: tfds.core.DatasetInfo) -> \
        dict:
    """
    Selects one random sample with all important information saved as dict.
    """
    sample = data.take(1)
    image_info = dict()
    for image, label in sample:
        image_info["shape"] = image.shape
        image_info["tensor"] = image
        image_info["content"] = show_image(image)
        image_info["label_tensor"] = label
        image_info["label_str"] = (
                ds_info
                .features["label"]
                .names[label.numpy()]
        )
        image_info["tensor_range"] = (
                tf.reduce_min(image),
                tf.reduce_max(image)
        )
    return image_info


def show_image(image_tensor: tf.Tensor):
    """
    Plots an image from a tensor.
    """
    image_numpy = image_tensor.numpy()
    plt.imshow(image_numpy)
    plt.axis('off')


def prepare_dataset(dataset: tf.data.Dataset, img_shape=512):
    """
    Dataset pipeline set-up. Preprocessing images with the preprocess_image \
            function, batching them and optimzing the pipeline via prefetching.
    """
    dataset = (
        dataset
        .map(functools.partial(preprocess_image, img_shape=img_shape), \
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(batch_size=128, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset


def preprocess_image(image: tf.Tensor, label: str, img_shape=512) \
        -> Tuple[tf.Tensor, str]:
    """
    Converts image tensor from uint8 (or any) to float32 tensor, \
    then reshapes to [img_shape, img_shape, colour_channels].
    """
    image = tf.image.resize(image, [img_shape, img_shape])
    return tf.cast(image, tf.float32), label


def create_tensorboard_callback(dir_name, model_name):
    """
    Creates a TensorBoard callback instance under \
            dir_name/model_name/current_datetime/ to store log files
    """
    log_dir = os.path.join(
            dir_name,
            model_name,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    return tensorboard_callback


def create_checkpoint_callback(checkpoint_path, monitor="val_acc"):
    """
    Creates a checkpoint callback (saves best model during training) \
            under checkpoint path
    """
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
            )
    return checkpoint_callback


def create_early_stopping_callback():
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1,
        mode='min',
        restore_best_weights=True,
        start_from_epoch=2
    )
    return early_stopping_callback

    
class AntiOverfittingCallback(tf.keras.callbacks.Callback): 
    """
    Halt the training when validation accuracy is \
        eps lower than the training accuracy
    """
    def __init__(self):
        self.eps = 0.07
        super().__init__()
            
    def on_epoch_end(self, epoch, logs={}): 
        val_acc = logs.get('val_sparse_categorical_accuracy')
        train_acc = logs.get('sparse_categorical_accuracy')
        if train_acc - val_acc > self.eps:
            print(f"val_acc {val_acc} is {self.eps} lower than acc {train_acc} \
                    -> stopping training at epoch {epoch}")   
            self.model.stop_training = True