import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Load and prepare the CIFAR-10 dataset
def load_real_samples():
    (trainX, _), (_, _) = cifar10.load_data()
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5
    return X


# Define the standalone discriminator model
def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# Define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 256 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


# Define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def generate_real_samples(dataset, n_samples):
    # Choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # Select images
    X = dataset[ix]
    # Generate class labels, 1 for real images
    y = np.ones((n_samples, 1))
    return X, y


def generate_fake_samples(generator, latent_dim, n_samples):
    # Generate points in latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # Reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    # Generate images
    X = generator.predict(x_input)
    # Create class labels, 0 for fake images
    y = np.zeros((n_samples, 1))
    return X, y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # Manually enumerate epochs
    for i in range(n_epochs):
        # Enumerate batches over the training set
        for j in range(bat_per_epo):
            # Get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # Update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # Generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # Update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # Prepare points in latent space as input for the generator
            X_gan = np.random.randn(latent_dim * n_batch)
            X_gan = X_gan.reshape(n_batch, latent_dim)
            # Create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # Update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # Summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # Evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# Function to save generated images for monitoring progress
def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples=150):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # Save plot
    save_plot(x_fake, epoch)
    # Save the generator model
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    generator.save(filename)


# Function to create and save a plot of generated images
def save_plot(examples, epoch, n=10):
    # Scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # Plot images
    for i in range(n):
        plt.subplot(2, 5, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    # Save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


# Size of the latent space
latent_dim = 100

# Create the discriminator
discriminator = define_discriminator()

# Create the generator
generator = define_generator(latent_dim)

# Create the GAN
gan_model = define_gan(generator, discriminator)

# Load and prepare CIFAR-10 training images
dataset = load_real_samples()

# Train the model
train(generator, discriminator, gan_model, dataset, latent_dim)
