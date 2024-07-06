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


def getvar(var: str, vars_file = os.path.join(".", ".credentials")) -> str:
    """
    Credentials, secrets fetching function.
    """
    with open(vars_file) as f:
        d = json.load(f)
        return d[var]


@timer
def load(dataset_name: str, dataset_path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
  """
  Loads the specified dataset from TensorFlow Datasets.
  """
  (train_data, test_data), ds_info = tfds.load(
      name=dataset_name,
      data_dir=dataset_path,
      split=["train", "validation"],
      shuffle_files=True,
      as_supervised=True,
      with_info=True
  )
  return train_data, test_data, ds_info


def random_sample(data: tf.data.Dataset, ds_info: tfds.core.DatasetInfo) -> dict:
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
        image_info["label_str"] = ds_info.features["label"].names[label.numpy()]
        image_info["tensor_range"] = (tf.reduce_min(image), tf.reduce_max(image))
    return image_info


def show_image(image_tensor: tf.Tensor):
  """
  Plots an image from a tensor.
  """
  image_numpy = image_tensor.numpy()
  plt.imshow(image_numpy)
  plt.axis('off')


def prepare_dataset(dataset: tf.data.Dataset):
    """
    Dataset pipeline set-up. Preprocessing images with the preprocess_image function, batching them and optimzing the pipeline via prefetching.
    """
    for subset in dataset:
      subset = (
          subset
          .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size=64)
          .prefetch(buffer_size=tf.data.AUTOTUNE)
      )


def preprocess_image(image: tf.Tensor, label: str, img_shape=224) -> Tuple[tf.Tensor, str]:
  """
  Converts image tensor from uint8 (or any) to float32 tensor, then reshapes to [img_shape, img_shape, colour_channels].
  """
  image = tf.image.resize(image, [img_shape, img_shape])
  return tf.cast(image, tf.float32), label


def create_tensorboard_callback(dir_name, model_name):
    """
    Creates a TensorBoard callback instance under dir_name/model_name/current_datetime/ to store log files
    """
    log_dir = os.path.join(dir_name, model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    print("Saving TensorBoard log files...")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"TensorBoard log files saved to {str(log_dir)}.")

    return tensorboard_callback

def create_checkpoint_callback(checkpoint_path, monitor="val_acc"):
    """
    Creates a checkpoint callback (saves best model during training) under checkpoint path
    """
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             monitor=monitor,
                                                             save_best_only=True,
                                                             verbose=1)
    return checkpoint_callback

def early_stopping_callbakc():
    return None
