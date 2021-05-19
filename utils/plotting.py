import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class HistoryPlotter():
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


class StandardizedDataset():
    """Returns a standard numpy output given tf.data.Dataset or list of numpy array
    The first dimension should be the number of examples, eg: (m,_,_,_). If tf.data.Dataset is
    a batch, then it should be (batches, batch_size, _, _)

    Arguments:
    dataset - should be tf.data.Dataset or [x,y] where x,y are numpy array
    mode - if a trained model is passed, it will be made to predict the 'dataset' values.
    """

    def __init__(self, dataset, model=None) -> None:

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

        if model is not None:
            if isinstance(model, tf.keras.Model):
                self.model = model
                self.pred = True
            else:
                raise TypeError("Model must be of tf.keras.Model class")
        else:
            self.pred = False

        self.dataset = dataset
        # print("Kind : ", self.kind)
        # print("prediction? :", self.pred)

    def generator(self):
        """A generator that returns x,y,y_pred* one by one.
        """
        # Numpy
        if self.kind == 'numpy':
            length = self.dataset[0].shape[0]

            # not tested prediction.
            if self.pred:
                for i in range(length):
                    prediction = self.model.predict(self.dataset[0][i])
                    yield self.dataset[0][i], self.dataset[1][i], prediction
            else:
                for i in range(length):
                    yield self.dataset[0][i], self.dataset[1][i]

        # TFDS without batches
        if self.kind == 'tfds':
            length = int(self.dataset.__len__())

            # models are trained using batches, thus when calling model.predict() it expects input of shape (batch_size,_,_)
            # thus not supporting prediction on non-batch tfds.
            for x, y in self.dataset.take(length):
                yield np.array(x), np.array(y)

        # TFDS with batches
        if self.kind == 'tfds_batch':
            num_batches = int(self.dataset.__len__())
            batch_size = int(self.dataset._batch_size)

            # tested and working
            if self.pred:
                for x_batch, y_batch in self.dataset.take(num_batches):
                    prediction = self.model.predict(x_batch)

                    for i in range(batch_size):
                        yield np.array(x_batch[i]), np.array(y_batch[i]), prediction[i]
            else:
                for x_batch, y_batch in self.dataset.take(num_batches):
                    for i in range(batch_size):
                        yield np.array(x_batch[i]), np.array(y_batch[i])


class ImagePlotter(StandardizedDataset):
    """Various ways to plot the images in a dataset.

    Arguments:
    dataset - should be tf.data.Dataset or [x,y] where x,y are numpy array
    """

    def __init__(self, dataset, model=None) -> None:
        super().__init__(dataset=dataset, model=model)

    def grid_plot(self, grid_size, figsize=(10, 16), hspace=0, wspace=0, cmap='binary', axis='off', title=None):
        """Plots the images in a grid

        Arguments:
        grid_size - a tuple of (rows, column)
        figsize - figure size
        hspace - vertical gap between each image
        wspace - horizontal gap between each image
        cmap - coloramp
        axis - to show x,y axis values
        title - title of each subplot; can be 'y_one_hot', 'y_pred_one_hot'
        """
        fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        gen = self.generator()

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):

                data = next(gen)  # can be x,y or x,y,y_pred
                data = list(data)

                # If image is of size h x w x 1 convert it to h x w
                x_shape = data[0].shape
                if len(x_shape) == 3 and x_shape[2] == 1:
                    data[0] = data[0].reshape(x_shape[0], x_shape[1])

                ax[i, j].imshow(data[0], cmap=cmap)
                ax[i, j].axis(axis)

                if title == "y_one_hot":
                    ax[i, j].set_title(np.argmax(data[1]), fontsize=15)
                elif title == "y_pred_one_hot":
                    ax[i, j].set_title(np.argmax(data[2]), fontsize=15)


if __name__ == '__main__':
    # history_dict = np.load('../00-LeNet/history.npy',
    #                      allow_pickle='TRUE').item()
    # hist = History(history_dict)
    # hist.loss(show_optimal=True)
    # hist.metric(show_optimal=True)

    # ---
    # original_x = np.ones(shape=(10, 10))
    # original_y = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    # predict_y = np.array([[11], [12], [13], [14], [15],
    #                      [16], [17], [18], [19], [20]])
    # d = tf.data.Dataset.from_tensor_slices((original_x, original_y)).batch(2)

    # p = StandardizedDataset(d, prediction=predict_y)

    # for test_x, test_y, pred_y in p.generator():
    #     print(test_x, test_y, pred_y)

    # ---
    # plotter = ImagePlotter(d)
    # plotter.grid_plot(grid_size=(2, 2))

    # ---
    from sklearn.preprocessing import OneHotEncoder
    model = tf.keras.models.load_model(
        "00-LeNet/run-LeNet-2021-05-19-16-21-01.h5")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshape x from 28x28 to 28x28x1
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Convert each element to an array; i.e., from (5000,) to (5000,1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(y_train)
    y_train = one_hot_encoder.transform(y_train)
    y_test = one_hot_encoder.transform(y_test)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def resize(image, label):
        """Resize the image to 32x32x1
        """
        image = tf.image.resize_with_pad(image, 32, 32, method='bilinear')
        # print(image.shape)
        return image, label
    train_ready = train.map(resize).batch(10)
    test_read = test.map(resize).batch(10)

    print("input shape: ", model.input_shape)
    plotter = ImagePlotter(train_ready, model)
    plotter.grid_plot(grid_size=(2, 2))
