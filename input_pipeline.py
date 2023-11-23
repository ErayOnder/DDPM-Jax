import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import jax.random as random

from diffusion_utils import forward_process


def get_dataset(dataset_name: str, split: str, shuffle: bool, batch_size: int, image_size: int):
    dataset = tfds.load(dataset_name, as_supervised=True, split=split, shuffle_files=shuffle)
    def preprocess(x, y):
        return tf.image.resize((tf.cast(x, tf.float32) / 127.5) - 1, (image_size, image_size))
    dataset = dataset.map(preprocess, tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

dataset_name = 'mnist'
split = 'train'
shuffle = True
batch_size = 64
image_size = 32
dataset = get_dataset(dataset_name, split, shuffle, batch_size, image_size)

def visualize_dataset(dataset, num_samples=5):
    images = next(iter(dataset.take(1)))

    rescaled_images = ((images.numpy() + 1) / 2.0 * 255.0).astype(np.uint8)

    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(rescaled_images[i], cmap='gray')
        plt.axis('off')
    plt.show()

visualize_dataset(dataset)

def visualize_noise(dataset):
    sample_image = next(iter(dataset))[0]

    for index, i in enumerate([1, 50, 100, 150, 200]):
        noisy_image, noise = forward_process(random.PRNGKey(0), sample_image)
        rescaled_image = ((noisy_image.numpy() + 1) / 2.0 * 255.0).astype(np.uint8)
        plt.subplot(1, 6, index + 1)
        plt.imshow(rescaled_image, cmap='gray')
        plt.axis('off')
    plt.show()

visualize_noise(dataset)
