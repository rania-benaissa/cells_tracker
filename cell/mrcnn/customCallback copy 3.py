import keras
import os
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()


# @tf.function
# def create_file(writer, LOG_DIR):
#     writer = tf.summary.create_file_writer(LOG_DIR)


class CustomCallback(keras.callbacks.Callback):

    def __init__(self, log_dir=None):

        self.LOG_DIR = log_dir

        # nb module
        self.nb_module = 6

        print("log dir :", self.LOG_DIR)

        self.loss_log = []

    @tf.function
    def init_summaries(self, logs=None):

        log_keys = list(logs.keys())

        for i in range(self.nb_module):
            # train event
            directory = self.LOG_DIR + "/" + \
                log_keys[i] + "/train_" + log_keys[i]

            self.loss_log.append(tf.summary.create_file_writer(directory))

            # validation event
            directory = self.LOG_DIR + "/" + \
                log_keys[i] + "/" + log_keys[i + self.nb_module]

            self.loss_log.append(tf.summary.create_file_writer(directory))

    @tf.function
    def add_train_scalar(self, name, value, i, step):
        # other model code would go here
        with self.loss_log[i * 2].as_default():
            tf.summary.scalar(name, value, step=step)

    @tf.function
    def add_val_scalar(self, name, value, i, step):
        # other model code would go here
        with self.loss_log[(i * 2) + 1].as_default():
            tf.summary.scalar(name, value, step=step)

    def add_scalars(self, logs, epoch):

        print("hey")
        tf.config.run_functions_eagerly(True)

        # names of the keys
        log_keys = list(logs.keys())

        for i in range(self.nb_module):
            print("hey2")

            # add train and validation scalars to a plot
            self.add_train_scalar(log_keys[i],
                                  logs[log_keys[i]], i, step=epoch)
            self.loss_log[i * 2].flush()
            print("hey3")

            self.add_val_scalar(
                log_keys[i], logs[log_keys[i + self.nb_module]], i, step=epoch)

            self.loss_log[(i * 2) + 1].flush()

            print("hey4 ", i)
        tf.config.run_functions_eagerly(False)

    def on_train_end(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):

        if(epoch == 0):

            self.init_summaries(logs)
        print("Done !")
        self.add_scalars(logs, epoch)
        print("hey5")

    def on_train_begin(self, logs=None):

        print("is executing eagerly :", tf.executing_eagerly())

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
    #     pass
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
