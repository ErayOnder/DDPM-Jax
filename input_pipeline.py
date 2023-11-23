
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def get_dataset():
    dataset = tfds.load('celeb_a', as_supervised=True, split='train')


