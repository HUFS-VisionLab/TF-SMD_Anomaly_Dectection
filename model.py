import os
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Conv2DTranspose, CuDNNLSTM, Bidirectional, Lambda, RepeatVector
from tensorflow.keras import backend as K

import tf_models

class DataLoader:
    def __init__(self, inputs_list, batch_size=None, is_shuffle=False, is_prefetch=False):
        def load_np_from_path(input_path):
            sequence = np.load(input_path.decode("utf-8"))

            return sequence.astype(np.float32)
        
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
    models_dict = {"Basic" : tf_models.Basic,
                   "AutoEncoder" : tf_models.Autoencoder,
                   "OneClass" : tf_models.OneClassAutoencoder
    }
    
    
    def __init__(self, config):
        self.config           = config
        self.learning_rate    = self.config.learning_rate
        self.beta_1           = self.config.beta_1
        self.beta_2           = self.config.beta_2
        self.epsilon          = self.config.epsilon
        self.epochs           = self.config.epochs
        self.save_path        = self.config.save_path

        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.config.inputs_dims], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
        
        self._build()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.saver = tf.train.Saver()
        
        
    def _build(self):
        model_type = self.config.model_type
        if '_' in model_type:
            model_type, option = model_type.split('_')
            self.config.model_type = model_type
            self.config.option = option
        else:
            self.config.option = None
        
        model = Model.models_dict[model_type]
        
        feed_list = [self.inputs, self.labels]
        outputs, loss_dict, feature_dict = model(feed_list, self.config)

        self.classifier_loss = loss_dict['classifier_loss']    
        self.recons_loss = loss_dict['recons_loss']
        self.loss = loss_dict['loss']
        
        
    def _calc_time(self, n_iteration, elasped_time):
        total_time = n_iteration * elasped_time

        hour = str(int(total_time // 3600)).zfill(2)
        minute = str(int((total_time % 3600) // 60)).zfill(2)
        second = str(int(total_time % 60)).zfill(2)

        
        log = f"{hour}:{minute}:{second}"

        return log
        
        
    def train(self, inputs_list):
        # Optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                beta1=self.beta_1, beta2=self.beta_2, epsilon=self.epsilon).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer()) # Initialize variables

        
        """ Training """
        trainData_loader = DataLoader(inputs_list, self.config.batch_size, is_shuffle=False, is_prefetch=False) # Dataset pipeline function
        n_iter = trainData_loader.batch_steps * self.epochs
        
        switch = 20
        self.loss_list = []
        check_time = lambda step: None if step != switch else time.time()
        best_recons_loss = 10000
        
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        for epoch in range(self.epochs):
            self.sess.run(trainData_loader.iterator.initializer)
            start      = time.time()
            total_loss = 0
            total_recons_loss = 0
            total_clf_loss = 0
            score = np.array([0,0]).astype(np.float32)
            
            
            for step  in range(trainData_loader.batch_steps):
                _start = check_time(step)

                
                batch_inputs = self.sess.run(trainData_loader.inputs)[0]
                batch_labels = np.array([[1,0] for i in range(batch_inputs.shape[0])]).astype(np.float32)
                
                if self.config.model_type != "OneClass":
                    request_list = [self.loss, self.optimizer]
                    loss, _ = self.sess.run(request_list, feed_dict={self.inputs:batch_inputs, self.labels:batch_labels})
                    recons_loss = loss
                    classifier_loss = 0
                    
                else:
                    request_list = [self.loss, self.recons_loss, self.classifier_loss, self.optimizer]
                    loss, recons_loss, classifier_loss, _ = self.sess.run(request_list, feed_dict={self.inputs:batch_inputs, self.labels:batch_labels})
                    
                total_loss += loss / trainData_loader.batch_steps
                total_recons_loss += recons_loss / trainData_loader.batch_steps
                total_clf_loss += classifier_loss / trainData_loader.batch_steps
                
                _end = check_time(step)
                if epoch == 0 and step == switch: print("Predict training time :", self._calc_time(n_iter-switch, _end-_start))
                    
            end = time.time()
            self.loss_list.append(total_recons_loss)
            
            log = None
            epoch_log = f"Epoch : {epoch}"
            time_log = f"Elasped time : {self._calc_time(1, end-start)}"
            
            if self.config.model_type != "OneClass":
                epoch_log = f"Epoch : {epoch}"
                loss_log = f"Recons loss : {total_loss:.4f}"
                
                log = f"{epoch_log}\t{loss_log}\t{time_log}"
            else:
                loss_log = f"Total loss : {total_loss:.4f}"
                recons_log = f"Recons loss : {total_recons_loss:.4f}"
                clf_log = f"Classifier loss : {total_clf_loss:.4f}"

                log = f"{epoch_log}\t{loss_log}\t{recons_log}\t{clf_log}\t{time_log}"
            
            print(log)
            
            if best_recons_loss > total_recons_loss:
                best_recons_loss = total_recons_loss
                if epoch >= self.epochs - 100:
                    self.saver.save(self.sess, '{}/model.ckpt'.format(self.save_path))
            
        print("Training completed")
        self.loss_list = np.array(self.loss_list)
        
        
    def inference(self, inputs_list):
        testData_loader = DataLoader(inputs_list, batch_size=1, is_shuffle=False, is_prefetch=False)
        self.sess.run(testData_loader.iterator.initializer)
        
        loss_list = []
        for step  in range(testData_loader.batch_steps):
            batch_inputs = self.sess.run(testData_loader.inputs)[0]
            recons_loss = self.sess.run(self.recons_loss, feed_dict={self.inputs:batch_inputs})
        
            loss_list.append(recons_loss)
            
        loss_list = np.array(loss_list)
        
        return loss_list
        
                                                                                 
    def load_weights(self):
        assert os.path.exists(self.save_path), "Model's checkpoint not found!"
        self.saver.restore(self.sess,'{}/model.ckpt'.format(self.save_path))