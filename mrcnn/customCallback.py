import keras
import torch
from torch.utils.tensorboard import SummaryWriter


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, log_dir=None, infos=None):

        self.LOG_DIR = log_dir

        self.run_infos = infos

        # nb loss
        self.nb_loss = 6

        self.train_losses = torch.zeros(infos["epochs"])
        self.val_losses = torch.zeros(infos["epochs"])

        self.train_loss_log = SummaryWriter(
            log_dir=self.LOG_DIR + '/train/')
        self.val_loss_log = SummaryWriter(log_dir=self.LOG_DIR + "/val/")

        # self.train_loss_log = SummaryWriter(
        #     log_dir=self.LOG_DIR, comment='train')
        # self.val_loss_log = SummaryWriter(log_dir=self.LOG_DIR, comment='val')

    def on_train_end(self, logs=None):

        train_metrics = {}
        val_metrics = {}

        train_metrics["best_loss"] = torch.min(self.train_losses)
        train_metrics["best_epoch_loss"] = torch.argmin(self.train_losses) + 1

        val_metrics["best_loss"] = torch.min(self.val_losses)
        val_metrics["best_epoch_loss"] = torch.argmin(self.val_losses) + 1

        self.train_loss_log.add_hparams(
            self.run_infos, train_metrics)
        self.val_loss_log.add_hparams(
            self.run_infos, val_metrics)

        self.train_loss_log.close()
        self.val_loss_log.close()

    def on_epoch_end(self, epoch, logs=None):

        # names of the keys
        log_keys = list(logs.keys())

        for i in range(self.nb_loss):

            # train
            self.train_loss_log.add_scalar(
                "Loss/" + log_keys[i], logs[log_keys[i]], epoch + 1)

            # add validation scalars to a plot
            self.val_loss_log.add_scalar(
                "Loss/" + log_keys[i], logs[log_keys[i + self.nb_loss]], epoch + 1)

        self.train_losses[epoch] = logs[log_keys[0]]
        self.val_losses[epoch] = logs[log_keys[6]]

    def on_train_begin(self, logs=None):

        pass
    # def on_epoch_begin(self, epoch, logs=None):
    #     pass

    # evaluation of the model using the validation set

    def on_test_begin(self, logs=None):
        # keys = list(logs.keys())
        # print("Start testing; got log keys: {}".format(keys))
        pass

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
