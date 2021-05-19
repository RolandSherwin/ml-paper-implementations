import numpy as np
import matplotlib.pyplot as plt


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
    """Returns a standard numpy output given tf.dataset input/ anyother input
    """

    def __init__(self, dataset=None) -> None:
        pass


if __name__ == '__main__':
    history_dict = np.load('../00-LeNet/history.npy',
                           allow_pickle='TRUE').item()
    hist = History(history_dict)
    hist.loss(show_optimal=True)
    # hist.metric(show_optimal=True)
