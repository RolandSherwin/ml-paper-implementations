from logging import exception
import numpy as np
from numpy.lib.arraysetops import isin
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow._api.v2 import data


class History():
    """Various ways to plot the training history
    """

    def __init__(self, history: dict) -> None:
        self.history = history

    def loss(self, val=True, show_optimal=False):
        """Plots the train_loss and val_loss (if specified)

        Arguments:
        val: Plots val_loss
        show_optimal: Draws a vertical dotted line at the lowest val_loss; works
            only if 'val' is True

        """

        plt.plot(self.history["loss"])
        plt.title('Model Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')

        if val:
            plt.plot(self.history["val_loss"])
            plt.legend(['train', 'val'], loc='upper right')

            if show_optimal:
                optimal_epoch = np.argmin(self.history['val_loss'])
                max_loss = max(np.max(self.history['loss']),
                               np.max(self.history['val_loss'])
                               )

                plt.plot([optimal_epoch, optimal_epoch],
                         [0, max_loss],
                         'g:',
                         lw=2)

        else:
            plt.legend(['train'], loc='upper right')
        plt.show()

    def metric(self, metric='accuracy', val=True, show_optimal=False):
        """Plots the progress of the metric (such as accuracy, recall).

        Arguments:
        metric: the metric to plot
        val: Plots the metric of the validation set
        show_optimal: Draws a vertical dotted line at the optimal val_metric; works
            only if 'val' is True
        """
        plt.plot(self.history[metric])
        plt.title('Model ' + metric.capitalize())
        plt.xlabel('epoch')
        plt.ylabel(metric)

        if val:
            plt.plot(self.history["val_" + metric])
            plt.legend(['train', 'val'], loc='lower right')

            if show_optimal:
                # currently works for metrics where high values are good.
                optimal_epoch = np.argmax(self.history['val_'+metric])
                min_metric = min(np.min(self.history[metric]),
                                 np.min(self.history['val_'+metric])
                                 )
                max_val_metric = np.max(self.history['val_'+metric])

                # vertical line
                plt.plot([optimal_epoch, optimal_epoch],
                         [min_metric, max_val_metric],
                         'g:',
                         lw=2)
                # horizontal line
                plt.plot([0, optimal_epoch],
                         [max_val_metric, max_val_metric],
                         'g:',
                         lw=2)

                # set marker on the optimal point
                # plt.plot([optimal_epoch],
                #          [max_val_metric],
                #          marker='o',
                #          markersize=5,
                #          color='green')

        else:
            plt.legend(['train'], loc='lower right')
        plt.show()


class PreprocessData():
    """Returns a standard numpy output given tf.data.Dataset or list of numpy array
    The first dimension should be the number of examples, eg: (m,_,_,_). If tf.data.Dataset is
    a batch, then it should be (batches, batch_size, _, _)

    Arguments:
    dataset - should be tf.data.Dataset or [x,y] where x,y are numpy array
    """

    def __init__(self, dataset) -> None:

        if isinstance(dataset, list) and isinstance(dataset[0], np.ndarray) and isinstance(dataset[1], np.ndarray):
            self.kind = 'numpy'
        elif isinstance(dataset, tf.data.Dataset):
            # check if it has batches
            try:
                dataset._batch_size
                self.kind = 'tfds_batch'
            except AttributeError:
                self.kind = 'tfds'
        else:
            raise TypeError(
                "Only pass in list of numpy array or tf.data.Dataset")

        self.dataset = dataset
        # print("Kind : " , self.kind)

    def generator(self):
        """A generator that returns x,y one by one; S
        """
        if self.kind == 'numpy':
            length = self.dataset[0].shape[0]

            for i in range(length):
                yield self.dataset[0][i], self.dataset[1][i]

        if self.kind == 'tfds':
            length = int(self.dataset.__len__())

            for x, y in self.dataset.take(length):
                yield x, y

        if self.kind == 'tfds_batch':
            num_batches = int(self.dataset.__len__())
            batch_size = int(self.dataset._batch_size)

            for x_batch, y_batch in self.dataset.take(num_batches):
                for i in range(batch_size):
                    yield x_batch[i], y_batch[i]


if __name__ == '__main__':
    # history_dict = np.load('../00-LeNet/history.npy',
    #                      allow_pickle='TRUE').item()
    # hist = History(history_dict)
    # hist.loss(show_optimal=True)
    # hist.metric(show_optimal=True)

    # ---
    original_x = np.ones(shape=(10, 10))
    original_y = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    d = tf.data.Dataset.from_tensor_slices((original_x, original_y)).batch(2)

    p = PreprocessData(d)

    for test_x, test_y in p.generator():
        print(test_x, test_y)
