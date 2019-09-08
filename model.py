import os
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import CuDNNLSTM, Bidirectional


class Dataset:
    def __init__(self, inputs_list, batch_size=None, is_shuffle=False, is_prefetch=False):
        def load_np_from_path(input_path):
            input_data = np.load(input_path.decode("utf-8"))

            return input_data.astype(np.float32)
        
        n_data = len(inputs_list)
        if batch_size is None : batch_size = n_data
            
        dataset = tf.data.Dataset.from_tensor_slices(inputs_list)
        dataset = dataset.map(map_func= lambda input: tuple(tf.py_func(load_np_from_path, inp=[input], Tout=[tf.float32])),
                              num_parallel_calls = 12)

        dataset = dataset.shuffle(buffer_size=10*batch_size) if is_shuffle != False else dataset
        dataset = dataset.prefetch(buffer_size=5*batch_size) if is_prefetch != False else dataset
        dataset = dataset.batch(batch_size)

        self.batch_steps = int(np.ceil(n_data / batch_size))
        self.iterator = dataset.make_initializable_iterator()
        self.inputs = self.iterator.get_next()


class Model:
    def __init__(self, config):
        self.get_list = lambda dir_path : sorted(glob.glob('{}/*'.format(dir_path))) if dir_path != None else None
        self.inputs_list      = self.get_list(config.inputPath)
        #self.units            = config.units
        self.n_layers         = config.n_layers
        self.is_bidirectioanl = config.is_bidirectional
        self.timesteps        = config.timesteps
        self.input_dim        = config.input_dim
        self.learning_rate    = config.learning_rate
        self.beta_1           = config.beta_1
        self.beta_2           = config.beta_2
        self.epsilon          = config.epsilon
        self.epochs           = config.epochs
        self.batch_size       = config.batch_size if config.batch_size is not None else len(self.inputs_list) 
        self.save_path        = config.save_path
        self.total_iter = (len(self.inputs_list) // self.batch_size) * self.epochs

        self.inputs = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.input_dim], name='inputs')
        
        self._build()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        
        
    def _build(self):
        self.x = self.inputs
        for i in range(self.n_layers-1):
            layer = CuDNNLSTM(units=self.input_dim, kernel_initializer='he_normal', return_sequences=True, name=f'layer_{i+1}')
            if self.is_bidirectioanl == 1:
                layer = Bidirectional(layer)
            
            self.x = layer(self.x)
        
        # last layer use only basic LSTM
        self.outputs = CuDNNLSTM(units=self.input_dim, kernel_initializer='he_normal', return_sequences=True, name=f'layer_{self.n_layers}')(self.x) 

        self.loss = tf.reduce_sum(tf.pow(self.outputs - self.inputs, 2))
        
        
    def _calc_time(self, n_iteration, elasped_time):
        total_time = n_iteration * elasped_time

        hour = total_time // 3600
        minute = (total_time % 3600) // 60

        log = "{} hour {} minute".format(hour, minute)

        return log
        
        
    def train(self):
        # Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                beta1=self.beta_1, beta2=self.beta_2, epsilon=self.epsilon).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer()) # Initialize variables

        
        """ Training """
        train_dataset = Dataset(self.inputs_list, self.batch_size, is_shuffle=True, is_prefetch=True) # Dataset pipeline function
        switch = 20
        self.loss_list = []
        check_time = lambda step: None if step != switch else time.time()
        
        for epoch in range(self.epochs):
            self.sess.run(train_dataset.iterator.initializer)

            start      = time.time()
            n_iter     = 0
            total_loss = 0
            
            for step  in range(train_dataset.batch_steps):
                _start = check_time(step)

                
                _batch_inputs = self.sess.run(train_dataset.inputs)[0]
                _loss, _      = self.sess.run([self.loss, self.optimizer], feed_dict={self.inputs:_batch_inputs})
                total_loss    += _loss / train_dataset.batch_steps

                
                _end = check_time(step)
                if epoch == 0 and step == switch: print("Predict training time :", self._calc_time(self.total_iter-switch, _end-_start))
            
            end = time.time()
            self.loss_list.append(total_loss)
            #print("Epoch :", epoch, ", Loss :", total_loss, 'Elasped time :', self._calc_time(1, end-start))
            
        
        print("Training completed")
        self.loss_list = np.array(self.loss_list)
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.saver.save(self.sess, '{}/model.ckpt'.format(self.save_path))
        print("Model saved")
                                                                                 
                                                                                 
    def inference(self, inputs_list):
        if inputs_list is not list : inputs_list = self.get_list(inputs_list)
            
        test_dataset = Dataset(inputs_list, batch_size=1, is_shuffle=False, is_prefetch=False)
        self.sess.run(test_dataset.iterator.initializer)
        
        loss_list = []
        for step  in range(test_dataset.batch_steps):
            _batch_inputs = self.sess.run(test_dataset.inputs)[0]
            _loss = self.sess.run(self.loss, feed_dict={self.inputs:_batch_inputs})
        
            loss_list.append(_loss)
            
        loss_list = np.array(loss_list)
        
        return loss_list
        
                                                                                 
    def load_weights(self):
        assert os.path.exists(self.save_path), "Model's checkpoint not found!"
        self.saver.restore(self.sess,'{}/model.ckpt'.format(self.save_path))
