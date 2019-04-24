import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()
import pickle
import zlib
import collections
import os


def deserialize(pz):
    if os.path.isfile(pz):
        compressed = open(pz, 'rb').read()
        pkl = zlib.decompress(compressed)
        tpl = pickle.loads(pkl)
        return tpl
    else:
        return []


def deserialize_non_compressed(pz):
    # compressed = open(pz).read()
    # pkl = zlib.decompress(compressed)
    tpl = pickle.loads(open(pz, 'rb').read())
    return tpl


def serialize(obj, fname):
    serialization_dir = os.path.dirname(fname)
    if not os.path.exists(serialization_dir):
        os.makedirs(serialization_dir)
    compressed = zlib.compress(pickle.dumps(obj))
    f = open(fname, 'wb')
    f.write(compressed)
    f.close()


class ResourceDictionary(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys.
       This is the base class for all resource dictionaries"""

    def __init__(self, negative=False, offset=1, *args, **kwargs):
        self.store = dict()
        self.inverse = dict()
        self.negative = negative
        self.offset = offset
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        if key in self.store:
            pass
        else:
            l = len(self.store) + self.offset
            if self.negative:
                l = -l
            self.store[key] = l
            self.inverse[l] = key

    def add(self, key):
        if key in self.store:
            return self.store[key]
        else:
            l = len(self.store) + self.offset
            if self.negative:
                l = -l
            self.store[key] = l
            self.inverse[l] = key
            return l

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def inverse(self):
        return self.inverse

    def copy(self):
        copy = ResourceDictionary()
        for k in self.store:
            copy[k] = self.store[k]
        return copy
