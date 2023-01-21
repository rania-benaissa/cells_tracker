import keras
import os
import tensorflow as tf


class CustomCallback(keras.callbacks.Callback):

    def __init__(self, log_dir=None):

        self.LOG_DIR = log_dir

        # nb module
        self.nb_module = 6

        self.loss_log = []

    def init_summaries(self, logs=None):

        log_keys = list(logs.keys())

        for i in range(self.nb_module):

            directory = self.LOG_DIR + "/" + \
                log_keys[i] + "/train_" + log_keys[i]

            # if not os.path.exists(directory):
            #     os.makedirs(directory)

            print(self.LOG_DIR)

            self.loss_log.append(
                tf.summary.create_file_writer(self.LOG_DIR))

            # directory = self.LOG_DIR + "/" + \
            #     log_keys[i] + "/" + log_keys[i + self.nb_module]

            # # if not os.path.exists(directory):
            # #     os.makedirs(directory)

            # print(directory)

            # self.loss_log.append(
            #     tf.summary.create_file_writer(directory))

    def on_train_end(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):

        if(epoch == 0):

            self.init_summaries(logs)

        # names of the keys
        log_keys = list(logs.keys())

        for i in range(self.nb_module):

            writer = self.loss_log[i * 2]

            # add train and validation scalars to a plot
            with writer.as_default():
                tf.summary.scalar(log_keys[i], logs[log_keys[i]], step=epoch)

                writer.flush()

            # writer = self.loss_log[(i * 2) + 1]

            # with writer.as_default():
            #     tf.summary.scalar(
            #         log_keys[i], logs[log_keys[i + self.nb_module]], step=epoch)

            #     writer.flush()

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    # def on_test_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start testing; got log keys: {}".format(keys))

    # def on_test_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop testing; got log keys: {}".format(keys))

    # def on_predict_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start predicting; got log keys: {}".format(keys))

    # def on_predict_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop predicting; got log keys: {}".format(keys))

    # def on_train_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_train_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    # def on_test_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_test_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    # def on_predict_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_predict_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
