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

        # Use reservoir sampling to drop or insert points
        self._tree_size = {}.get('tree_size', 256)
        self._shingle_size = {}.get('shingle_size', 6)
        self._codisp_threshold = 80
        self._codisp_list = []
        # self._points_array = np.arange(self._tree_size)
        self._point_idx = 0
        self._shingle_points = []

        # Create a forest of empty trees
        self._forest = []
        for _ in range(self._num_trees):
            tree = rrcf.RCTree()
            self._forest.append(tree)

    def handleRecord(self, inputData):
        # For each tree in the forest...
        avg_codisp = 0
        point = inputData['value']
        self._shingle_points.append(point)
        if len(self._shingle_points) > self._shingle_size:
            self._shingle_points.pop(0)
        else:
            return (0,)

        tree_count = 0
        for tree in self._forest:
            # If tree is above permitted size...
            k = random.randint(0, 1)
            if k > 0:
                tree_count+=1
                point_idx = len(tree.leaves)
                if len(tree.leaves) >= self._tree_size:
                    # Insert the new point into the tree
                    point_idx = int(random.random() * self._tree_size)
                    tree.forget_point(index=point_idx)
                tree.insert_point(self._shingle_points, index=point_idx)
                avg_codisp += tree.codisp(point_idx) / tree_count

        # self._point_idx+=1
        # self._point_idx=self._point_idx%self._tree_size

        # self._codisp_list.append(avg_codisp)
        # if len(self._codisp_list) > self._shingle_size:
        #     self._codisp_list.pop(0)
        #
        # qt = np.quantile(self._codisp_list, 0.999)
        # self._codisp_threshold = qt

        # result = np.quantile(self._codisp_list, 0.99)
        #
        result_num = 1-1/(1+np.log(1+avg_codisp))
        #result = result_num if avg_codisp > qt else 0
        #result = avg_codisp > self._codisp_threshold

        return (result_num,)
