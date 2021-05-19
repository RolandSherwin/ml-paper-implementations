import tensorflow.keras as keras
import os
import time


class Callbacks():
    """Contains various wrapped Keras callbacks
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.root_logdir = os.path.join(os.curdir, "logs")

    def run_id(self, string=None):
        """Returns the current run_id and the run_logdir

        Arguments:
        string: Adds that string to run_id as "run-your_string-time.."
        """
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        if string:
            run_id = "run-" + string + "-" + current_time
        else:
            run_id = 'run-' + current_time

        return run_id, os.path.join(self.root_logdir, run_id)

    def TensorBoard(self, folder_name=None, **kwargs) -> keras.callbacks.TensorBoard:
        """Returns TensorBoard callback with folder name as 'run-folder_name-time..'
        and with the **kwargs passed to TensorBoard object
        """
        run_id, run_logdir = self.run_id(folder_name)

        return keras.callbacks.TensorBoard(run_logdir, **kwargs)

    def ModelCheckpoint(self, file_name=None, **kwargs) -> keras.callbacks.ModelCheckpoint:
        """Returns ModelCheckpoint callback with filename as 'run-file_name-time..'
        and with the **kwargs passed to ModelCheckpoint object
        """
        run_id, run_logdir = self.run_id(file_name)

        return keras.callbacks.ModelCheckpoint(run_id+".h5", **kwargs)


if __name__ == "__main__":

    cb = Callbacks()
    print(cb.run_id())
    print(type(cb.TensorBoard()))
    print(type(cb.ModelCheckpoint()))
