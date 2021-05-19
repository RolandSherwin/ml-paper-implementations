import matplotlib.pyplot as plt


class History():
    """Various ways to plot the training history
    """

    def __init__(self, history: dict) -> None:
        self.history = history

    def loss(self, val=True):
        """Plots the train_loss and val_loss (if specified)
        """
        print("GG")
        plt.plot(self.history["loss"])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        if val:
            plt.plot(self.history["val_loss"])
            plt.legend(['train', 'val'], loc='upper right')
        else:
            plt.legend(['train'], loc='upper right')
        plt.show()

    def metric(self, metric='accuracy', val=True):
        plt.plot(self.history[metric])
        plt.title('Model ' + metric.capitalize())
        plt.xlabel('epoch')
        plt.ylabel(metric)

        if val:
            plt.plot(self.history["val_" + metric])
            plt.legend(['train', 'val'], loc='lower right')
        else:
            plt.legend(['train'], loc='lower right')
        plt.show()
