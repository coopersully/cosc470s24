import time

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers

import os
from imghdr import what

IMAGE_DIRECTORY = 'flickr-dataset/'


# Load and preprocess dataset
def load_data(directory):
    # Load images using image_dataset_from_directory
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        label_mode=None,  # only interested in the images (unsupervised)
        image_size=(32, 32),
        batch_size=32,
        shuffle=True
    )
    # Normalize images to [-1, 1] to match the `tanh` activation in the generator's output
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)
    return dataset


# Generator model
def make_generator_model():
    model = tf.keras.Sequential([
        # Use the Input layer to specify the input shape
        tf.keras.layers.Input(shape=(100,)),

        layers.Dense(8 * 8 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


# Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


# Defining the loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer,
                       image_batch.shape[0], noise_dim)

        # Produce images for the GIF
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


# For gradient tape
@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# Helper function to make sure image directory is right bc sometimes I'm dumb
def verify_image_directory(directory_path, min_image_count=100):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Check if the directory contains at least min_image_count images
    image_extensions = {'jpeg', 'png', 'gif', 'bmp', 'tiff'}
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    valid_image_count = 0

    for image_file in image_files:
        # Check if the file is an image
        file_path = os.path.join(directory_path, image_file)
        if what(file_path) in image_extensions:
            valid_image_count += 1
        else:
            raise ValueError(f"Non-image file found in the directory: {image_file}")

    if valid_image_count < min_image_count:
        raise ValueError(
            f"The directory contains less than {min_image_count} images. Found: {valid_image_count} images.")

    print(f"Directory verified: {directory_path} contains {valid_image_count} valid image files.")


# Constants
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

if __name__ == '__main__':
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    verify_image_directory(IMAGE_DIRECTORY)
    train_dataset = load_data(IMAGE_DIRECTORY)

    train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer)
