import tensorflow as tf
import os
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave

def sample_Z(m,n):
    return np.random.uniform(-1., 1., size=[m,n])

def get_y(x):
    return 10 + x*x;

def sample_data(n=10000, scale=100):
    data = []
    x = scale*(np.random.random_sample((n,))-0.5)
    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])
    return np.array(data)

def _generator(Z, hsize=[16,16]):
    # It should generate an image of size same as actual image
    with tf.variable_scope("GAN/Generator", reuse=False):
        h1 = tf.layers.dense(Z,hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,2)
    return out

def _discriminator(X, hsize=[16,16], reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        out = tf.layers.dense(h3,1)
    return out, h3

image_dim = 784 #28*28
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1./tf.sqrt(shape[0]/2.))

def generator(Z):
    # Input : array of size 100
    # It should generate an image of size same as actual image
    with tf.variable_scope("GAN/Generator", reuse=False):
        hidden_layer =tf.matmul(Z,tf.Variable(glorot_init([noise_dim, gen_hidden_dim])))
        hidden_layer = tf.add(hidden_layer, tf.Variable(tf.zeros([gen_hidden_dim])) )
        hidden_layer = tf.nn.relu(hidden_layer)


        hidden_layer2 = tf.matmul(hidden_layer,tf.Variable(glorot_init([gen_hidden_dim, gen_hidden_dim])))
        hidden_layer2 = tf.add(hidden_layer2, tf.Variable(tf.zeros([gen_hidden_dim])) )
        hidden_layer2 = tf.nn.relu(hidden_layer2)

        hidden_layer3 = tf.matmul(hidden_layer2,tf.Variable(glorot_init([gen_hidden_dim, gen_hidden_dim])))
        hidden_layer3 = tf.add(hidden_layer3, tf.Variable(tf.zeros([gen_hidden_dim])) )
        hidden_layer3 = tf.nn.relu(hidden_layer3)

        hidden_layer4 = tf.matmul(hidden_layer3,tf.Variable(glorot_init([gen_hidden_dim, gen_hidden_dim])))
        hidden_layer4 = tf.add(hidden_layer4, tf.Variable(tf.zeros([gen_hidden_dim])) )
        hidden_layer4 = tf.nn.relu(hidden_layer4)

        out_layer = tf.matmul(hidden_layer4, 
            tf.Variable(glorot_init([gen_hidden_dim, image_dim])))
        out_layer = tf.add(out_layer, tf.Variable(tf.zeros([image_dim])) )
        out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def discriminator(X, hsize=[16,16], reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        #h1 = tf.layers.dense(X, hsize[0],activation=tf.nn.leaky_relu)
        #h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        #h3 = tf.layers.dense(h2,2)
        #out = tf.layers.dense(h3,1)
        hidden_layer=tf.matmul(X,tf.Variable(glorot_init([image_dim, disc_hidden_dim])))
        hidden_layer = tf.add(hidden_layer, tf.Variable(tf.zeros([disc_hidden_dim])) )
        hidden_layer = tf.nn.relu(hidden_layer)

        hidden_layer2 = tf.matmul(hidden_layer,tf.Variable(glorot_init([disc_hidden_dim, disc_hidden_dim])))
        hidden_layer2 = tf.add(hidden_layer2, tf.Variable(tf.zeros([disc_hidden_dim])) )
        hidden_layer2 = tf.nn.relu(hidden_layer2)

        hidden_layer3 = tf.matmul(hidden_layer2,tf.Variable(glorot_init([disc_hidden_dim, 2])))
        hidden_layer3 = tf.add(hidden_layer3, tf.Variable(tf.zeros([2])) )
        hidden_layer3 = tf.nn.relu(hidden_layer3)

        out_layer = tf.matmul(hidden_layer3, 
            tf.Variable(glorot_init([2, 1])))
        out_layer = tf.add(out_layer, tf.Variable(tf.zeros([1])) )
        out_layer = tf.nn.sigmoid(out_layer)
    return out_layer, hidden_layer2

def load_data(directory_path):
    print(len(os.listdir(directory_path)))
    train_data = []
    for file_ in sorted(os.listdir(directory_path)):
        if file_.endswith('.png'):
            if directory_path.endswith('/'):
                 image_path = directory_path + file_
            else: image_path = directory_path + '/' + file_

            image = imread(image_path)/255.0 # Normalize values
            image = np.expand_dims(image, axis=-1) # Add channel dim
            train_data.append(image)
    train_data = np.array(train_data)
    return train_data

def main():
    print("Hello World from RaGAN")

    Z = tf.placeholder(tf.float32, [None,noise_dim])
    X = tf.placeholder(tf.float32, [None,image_dim])

    G_sample = generator(Z )
    r_logits, r_rep = discriminator(X)
    f_logits, g_rep = discriminator(G_sample, reuse=True)

    # Loss functions
    disc_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=r_logits, labels=tf.ones_like(r_logits)) + 
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=f_logits, labels=tf.zeros_like(f_logits))) 

    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=f_logits, labels=tf.ones_like(f_logits)))

    # Optimizers
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        scope="GAN/Discriminator")

    gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, 
        var_list=gen_vars) # G train step
    disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, 
        var_list=disc_vars) # G train step

    #Session
    config = tf.ConfigProto(device_count = {'GPU': 2})
    sess = tf.Session(config = config)
    tf.global_variables_initializer().run(session=sess)

    # Training the network
    batch_size=227
    nd_steps = 10
    ng_steps = 10
    for i in range(1000):
        # Reading the data
        X_batch = load_data("data/Hernia"); 

        X_batch = np.reshape(X_batch, (227, 28*28))

        Z_batch = glorot_init([batch_size, noise_dim ]); 

        Z_batch = Z_batch.eval(session=sess)

        for _ in range(nd_steps):
            _ ,dloss = sess.run([disc_step, disc_loss], 
                feed_dict={X:X_batch, Z:Z_batch})

        rrep_dstep, grep_dstep  = sess.run([r_rep, g_rep], 
            feed_dict={X:X_batch, Z:Z_batch})

        for _ in range(ng_steps):
            _ ,gloss = sess.run([gen_step, gen_loss], 
                feed_dict={Z:Z_batch})

        rrep_gstep, grep_gstep  = sess.run([r_rep, g_rep], 
            feed_dict={X:X_batch, Z:Z_batch})
        print("Iteration: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i, dloss, gloss))

    Z_test = glorot_init([1, noise_dim])
    Z_test = Z_test.eval(session = sess) #glorot_init([1, noise_dim])
    #print(Z_test) #image_test = generator(Z)
    #print(sess.run(G_sample, feed_dict={Z:Z_test}) )
    gen_out = sess.run(G_sample, feed_dict={Z:Z_test}) 
    imgnp = np.reshape(gen_out, (28, 28))
    imsave('img.png', imgnp) 

main();
