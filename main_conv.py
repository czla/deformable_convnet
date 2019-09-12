from ops import print_vars, mnistloader, conv2d, deform_conv2d, fc_layer, softmax_cross_entropy
from config import DeformConvConfig, SAVE_DIR
from utils import create_dir
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import logging
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(level=logging.DEBUG, format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('./main_conv.log'))

class Conv(DeformConvConfig):
    def __init__(self, dataset = 'MNIST'):
        logger.info("Initialization begins")
        DeformConvConfig.__init__(self)
        logger.info("MNIST dataset load begins")
        self.train_data, self.test_data, self.val_data = mnistloader("../MNIST_data/")
        logger.info("MNIST dataset load done...")

        self.image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 784])
        self.label = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 10])
        self.label_logit = self.build_model(self.image)
        self.sess=tf.Session()
        self.cross_entropy = softmax_cross_entropy(logits=self.label_logit, labels= self.label)
        self.run_train = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.label, axis=1), tf.argmax(self.label_logit, axis=1)),tf.float32))
 
    def build_model(self, x):
        logger.info("Buidling model starts...")
        x_r = tf.reshape(x, [-1, 28, 28, 1])
        # o = deform_conv2d(x_r, [7,7,1,50], [5,5,1,32], activation=tf.nn.relu, scope="deform_conv1")
        o = conv2d(x_r, [5, 5, 1, 32], padding=True, activation=tf.nn.relu, scope="conv1")
        o = tf.nn.max_pool(o, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        # o = deform_conv2d(o, [7,7,32,50], [5,5,32,64], activation=tf.nn.relu, scope="deform_conv2")
        o = conv2d(o, [5, 5, 32, 64], padding=True, activation=tf.nn.relu, scope="conv2")
        o = tf.nn.max_pool(o, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        o = tf.reshape(o, [self.batch_size, -1])
        o = fc_layer(o, 512, activation=tf.nn.relu, scope="fc1")
        o = fc_layer(o, 10, scope="fc2") 
        print_vars("trainable_variables")
        logger.info("Buidling model done")
        return o
    
    def initialize(self):
    	self.sess.run(tf.global_variables_initializer())

    def restore(self):
        logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(SAVE_DIR))
        logger.info("Restoring model done.")

    def train(self):
        create_dir(SAVE_DIR)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 10)
        ntrain = len(self.train_data.image)
        nbatch = ntrain//self.batch_size

        for epoch in range(self.epoch):
            # shuffle start
            index = np.arange(ntrain)
            np.random.shuffle(index)
            shuffle_image = self.train_data.image[index]
            shuffle_label = self.train_data.label[index]
            # shuffle end
            epoch_accuracy = 0
            test_accuracy = 0
            start_time = time.time()
            for batch in tqdm(range(nbatch), ascii = True, desc = "batch"):
                start = self.batch_size*batch
                end = self.batch_size*(batch+1)
                train_feed_dict = {self.image : shuffle_image[start:end], self.label : shuffle_label[start:end]}
                _, batch_accuracy = self.sess.run([self.run_train, self.accuracy], feed_dict = train_feed_dict)
                epoch_accuracy += batch_accuracy

            end_time = time.time()
            epoch_accuracy/=nbatch
            print('\nInference time_train_100_28_28_1:{} ms '.format((end_time - start_time)/nbatch))

            index2 = np.arange(len(self.test_data.image))
            np.random.shuffle(index2)
            shuffle_image_test = self.test_data.image[index2]
            shuffle_label_test = self.test_data.label[index2]
            batch_test = len(self.test_data.image)//self.batch_size
            start_time = time.time()
            for batch in tqdm(range(batch_test), ascii = True, desc = "batch"):
                start = self.batch_size*batch
                end = self.batch_size*(batch+1)
                test_feed_dict = {self.image : shuffle_image_test[start:end], self.label : shuffle_label_test[start:end]}
                batch_accuracy = self.sess.run( self.accuracy, feed_dict = test_feed_dict)
                test_accuracy += batch_accuracy

            end_time = time.time()
            test_accuracy/=batch_test
            print('\nInference time_test_100_28_28_1:{} ms '.format((end_time - start_time)/batch_test))
            # test_feed_dict = {self.image : self.test_data.image, self.label : self.test_data.label}
            # test_accuracy = self.sess.run(self.accuracy, feed_dict=test_feed_dict)
            logger.info("Epoch({}/{}) train_accuracy : {}, test_accuracy : {}".format(epoch+1, self.epoch, epoch_accuracy, test_accuracy))
            if epoch%self.save_every == self.save_every-1:
                saver.save(self.sess, os.path.join(SAVE_DIR, 'model'), global_step = epoch+1)

if __name__ == '__main__':
    model = Conv()
    model.initialize()
    model.train()
