from nab.detectors.base import AnomalyDetector
import random
import numpy as np
import rrcf
import math


def sigmoid(x):
    result = 1 / (1 + math.exp(-x))
    return result


class RrctDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(RrctDetector, self).__init__(*args, **kwargs)
        self._num_trees = {}.get('num_trees', 40)
        self._shingle_size = {}.get('shingle_size', 6)

        # Use reservoir sampling to drop or insert points
        self._tree_size = {}.get('tree_size', 256)
        self._codisp_threshold = 0
        self._codisp_array = np.zeros(self._tree_size)
        # self._points_array = np.arange(self._tree_size)
        # self._point_idx = 0

        # Create a forest of empty trees
        self._forest = []
        for _ in range(self._num_trees):
            tree = rrcf.RCTree()
            self._forest.append(tree)

    def handleRecord(self, inputData):
        # For each tree in the forest...
        avg_codisp = 0
        timestamp = inputData['timestamp']
        point = inputData['value']
        point_index = len(self._forest[0].leaves)

        if point_index >= self._tree_size:
            point_index = int(random.random() * self._tree_size)

        for tree in self._forest:
            # If tree is above permitted size...
            if len(tree.leaves) < self._tree_size:
                # Insert the new point into the tree
                tree.insert_point(point, index=point_index)
            else:
                # Drop the sampled point for that index and insert a new sample
                tree.forget_point(point_index)

                # Insert the new point into the tree
                tree.insert_point(point, index=point_index)

                # drop the oldest point
                #self._point_idx += 1
                #self._point_idx = self._point_idx % self._tree_size
                # self._points_array = np.roll(self._points_array, -1)


            new_codisp = tree.codisp(point_index)
            avg_codisp += new_codisp / self._num_trees

        self._codisp_array[point_index] = avg_codisp
        self._codisp_threshold = np.quantile(self._codisp_array, 0.99)

        # if avg_codisp > self._codisp_threshold + 50:
        #     self._codisp_threshold = np.quantile(self._codisp_array, 0.99)

        if avg_codisp > self._codisp_threshold:
            result = 1
        else:
            result = 0
        return (result, )