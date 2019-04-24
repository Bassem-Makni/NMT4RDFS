import csv
import os
import time
from collections import Iterable
from collections import OrderedDict

import keras.backend as K
import numpy as np
import six
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split


def train_validate_test_split(df, train_percent=0.6, validate_percent=0.2, stratify=None, seed=1):
    val_test_percent = 1 - train_percent
    test_percent = (1 - (train_percent + validate_percent))
    test_percent = test_percent / (test_percent + validate_percent)
    if stratify:
        df_train, df_val_test = train_test_split(df, test_size=val_test_percent, random_state=seed,
                                                 stratify=df[stratify])
        df_val, df_test = train_test_split(df_val_test, test_size=test_percent, random_state=seed,
                                           stratify=df_val_test[stratify])
    else:
        df_train, df_val_test = train_test_split(df, test_size=val_test_percent, random_state=seed)
        df_val, df_test = train_test_split(df_val_test, test_size=test_percent, random_state=seed)
    return df_train, df_val, df_test


def true_acc2(y_true, y_pred):
    return K.mean(K.all(K.equal(y_true, K.round(y_pred)), axis=-1))


def true_acc(y_true, y_pred):
    """
        All Accuracy
        https://github.com/rasmusbergpalm/normalization/blob/master/train.py#L10
    """
    return K.mean(
        K.all(
            K.equal(
                K.argmax(y_true, axis=-1),
                K.argmax(y_pred, axis=-1)
            ),
            axis=1)
    )


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype=np.int)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.bool)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class CSVLoggerTimed(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        self.starttime = time.time()
        super(CSVLoggerTimed, self).__init__()

    def on_train_begin(self, logs=None):
        self.starttime = time.time()
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['time'] = time.time()
        logs['time_delta'] = time.time() - self.starttime

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
