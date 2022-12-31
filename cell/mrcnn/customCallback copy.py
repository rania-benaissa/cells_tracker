import keras
import os
from torch.utils.tensorboard import SummaryWriter


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, log_dir=None):

        self.LOG_DIR = log_dir

        # nb module
        self.nb_module = 6

        self.loss_log = [SummaryWriter(log_dir=self.LOG_DIR)
                         for i in range(self.nb_module)]

    def on_train_end(self, logs=None):

        for i in range(self.nb_module):
            self.loss_log[i].close()

    def on_epoch_end(self, epoch, logs=None):

        # names of the keys
        log_keys = list(logs.keys())

        for i in range(self.nb_module):

            # self.loss_log.add_scalars('loss', {'train': logs["loss"],
            #                                    'val': logs["val_loss"]}, epoch)

            # add train and validation scalars to a plot
            self.loss_log[i].add_scalars(log_keys[i], {"train_" + log_keys[i]: logs[log_keys[i]],
                                                       log_keys[i + self.nb_module]: logs[log_keys[i + self.nb_module]]}, epoch)

    def on_train_begin(self, logs=None):

        pass
    # def on_epoch_begin(self, epoch, logs=None):
    #     pass
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
