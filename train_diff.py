import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
import sys
import math

from data import dataloader
import models
import training_sessions

print(tf.config.list_physical_devices("GPU"))

max_epoch = 5000
store_img_iter = 300
display_stats_iter = 400
batch_size = 48
norm_groups = 8
learning_rate = 1e-4

img_size = 64
n_channels = 3
image_shape = (img_size, img_size, n_channels)

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2

timesteps = 500
b1, b2 = 0.0001, 0.02
beta = np.cos(np.linspace(0, math.pi / 2, timesteps)) * (b2 - b1) + b1
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1 - alpha_bar)

def forward_noise(x, t):
    noise = tf.random.normal(shape=x.shape)

    mean = sqrt_alpha_bar[t] * x
    noised_image = mean + one_minus_sqrt_alpha_bar[t] * noise

    return noised_image, noise

def get_samples(n_samples):
    idx = tf.random.uniform([n_samples], 0, dataloader.data_size, dtype=tf.dtypes.int32)
    timestamps = tf.random.uniform([n_samples], 0, timesteps, dtype=tf.dtypes.int32)
    images = []
    noises = []
    for i in range(n_samples):
        image, noise = forward_noise(dataloader.get_batch(idx[i], 1)[0], timestamps[i])
        images.append(image)
        noises.append(noise)
    
    return (tf.convert_to_tensor(images), np.array(tf.expand_dims(timestamps, -1))), tf.convert_to_tensor(noises)

def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss

def ddpm(x_t, pred_noise, t, generator):
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / ((1 - alpha_t_bar) ** .5)
    mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)
    
    var = np.take(beta, t)
    if t == 0:
        z = tf.zeros(x_t.shape)
    else:
        z = generator.normal(shape=x_t.shape)

    return mean + (var ** .5) * z

def generate_images(n_images, model, seed):
    gen = tf.random.Generator.from_seed(seed)
    return generate_images(n_images, model, gen)

def generate_images(n_images, model, gen):
    images = gen.normal((n_images, image_shape[0], image_shape[1], image_shape[2]))

    for t in reversed(range(timesteps)):
        noise = model.call([images, tf.convert_to_tensor(np.array([t] * n_images))])
        
        alpha_t = np.take(alpha, t)
        alpha_t_bar = np.take(alpha_bar, t)

        eps_coef = (1 - alpha_t) / ((1 - alpha_t_bar) ** .5)
        mean = (1 / (alpha_t ** .5)) * (images - eps_coef * noise)
    
        var = 1 - (alpha_t_bar[max(t - 1, 0)]) / (1 - alpha_t_bar)

        if t == 0:
            z = tf.zeros(images.shape)
        else:
            z = gen.normal(shape=images.shape)
        
        images = mean + (var ** .5) * z
    images = tf.clip_by_value(images, -1, 1)
    return images

def save_model(model):
    model.save(os.path.join(os.path.dirname(__file__), "trained_models/diffusion"))

def train(model, sess):
    ma_loss = 0
    n = 0
    for epoch in range(max_epoch):
        n_batches = int(dataloader.data_size / batch_size)
        for batch in range(n_batches):
            X, Y = get_samples(batch_size)

            loss = model.train_on_batch(X, Y)
            ma_loss += loss
            n += 1

            if (batch % display_stats_iter == 0) and (batch != 0):
                print(f"{epoch}: {batch}/{n_batches}) loss = {loss}, ma_loss = {ma_loss / n}")
                ma_loss = 0; n = 0
                images = generate_images(4, model, 72)
                sess.save_plot(images, 2)
                sess.save()

def main():
    model = None
    group = training_sessions.SessionGroup("D")
    sess = None
    if "resume" in sys.argv:
        sess = group.load_sess(path=group.latest())
        model = sess.models["diff"]
    else:
        model = models.define_noise_predictor(image_shape, learning_rate, n_layers=6)
        sess = group.new_sess(models={"diff" : model})
    train(model, sess)

if __name__ == "__main__":
    main()