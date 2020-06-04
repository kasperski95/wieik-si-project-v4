from tensorflow.keras.callbacks import Callback


class SaveCallback(Callback):
    def __init__(self, name: str, mod: int):
        Callback.__init__(self)
        self._ctr = 0
        self._name = name
        self._n_saved = 0
        self._mod = mod

    def on_epoch_end(self, batch, logs={}):
        self._ctr += 1
        if self._ctr == self._mod:
            self._ctr = 0
        else:
            return

        self.model.save(f"checkpoints/{self._name}_{self._n_saved:05}-epochs.h5")
        self._n_saved += 1
