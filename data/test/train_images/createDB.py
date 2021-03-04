#!/usr/bin/env python3
# https://www.tensorflow.org/tutorials/load_data/images

import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pathlib


print(tf.__version__)

data_dir = pathlib.Path(__file__).parent.absolute()
print(data_dir)

image_count = len(list(data_dir.glob('*/*crop.png')))
print(image_count)

roses = list(data_dir.glob('Albi-Himbeer-Maracuja/*'))
# PIL.Image.open(str(roses[0])).show()

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

from tensorflow.keras import layers

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = 12

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1
    )

os.makedirs('./model', exist_ok=True)

model.save('model', save_format='tf')

########################################################################

if False:
    ###################################################
    # HARD MAYBE#
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1
    )

    model.save('model', save_format='tf')

    all_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        all_saver.save(sess, dir + '/data-all')

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors,
    # and stored with the default serving key
    import tempfile

    # MODEL_DIR = data_dir #tempfile.gettempdir()
    # version = 1
    # export_path = os.path.join(MODEL_DIR, str(version))
    # print('export_path = {}\n'.format(export_path))

    # tf.keras.models.save_model(
    #    model,
    #    export_path,
    #    overwrite=True,
    #    include_optimizer=True,
    #    save_format=None,
    #    signatures=None,
    #    options=None
    # )

    # print('\nSaved model:')
