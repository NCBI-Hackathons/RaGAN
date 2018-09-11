import tensorflow as tf
import numpy as np


def get_y(x):
    return 10 + x*x;

def sample_data(n=10000, scale=100):
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)

def generator(Z, hsize=[16,16]):
    with tf.variable_scope("GAN/Discriminator", reuse=False):
        h1 = tf.layers.dense(Z,hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,2)
    return out

"""
def discriminator(X, hsize=[16,16]):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers
"""

def main():
    print("Hello World from RaGAN")
    a = sample_data()
    Z = tf.placeholder(tf.float32, [None,2])
    G_sample = generator(Z )

main();
