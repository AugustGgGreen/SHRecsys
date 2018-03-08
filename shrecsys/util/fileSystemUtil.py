import os
import pickle
class FileSystemUtil(object):
    def make_dir(self, path):
        """ Create a directory if there isn't one already. """
        try:
            os.mkdir(path)
        except OSError:
            pass

    def save_obj(self, obj, root, name):
        with open(os.path.join(root, name + '.pkl'), 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, root, name):
        with open(os.path.join(root, name + '.pkl'), 'rb') as f:
            return pickle.load(f)