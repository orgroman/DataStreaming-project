from collections import deque

from nab.detectors.base import AnomalyDetector
import random
import numpy as np
import rrcf
import math


def sigmoid(x):
    result = 1 / (1 + math.exp(-x))
    return result


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


class RrctDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(RrctDetector, self).__init__(*args, **kwargs)
        self._num_trees = {}.get('num_trees', 40)

        # Use reservoir sampling to drop or insert points
        self._tree_size = {}.get('tree_size', 256)
        self._shingle_size = {}.get('shingle_size', 4)
        self._codisp_threshold = 80
        self._codisp_list = []
        # self._points_array = np.arange(self._tree_size)
        self._point_idx = 0
        self._shingle_points = deque()
        self._timestamp_idx = deque()

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
        self._shingle_points.append(point)
        if len(self._shingle_points) > self._shingle_size:
            self._shingle_points.popleft()
        else:
            return (0,)

        tree_count = 0
        codisp_list = []

        forget_point = None
        self._timestamp_idx.append(self._point_idx)
        if len(self._timestamp_idx) > self._tree_size:
            forget_point = self._timestamp_idx.popleft()

        # forget_point = None
        # if len(self._timestamp_idx) > self._tree_size:
        #     forget_point = self._timestamp_idx.popleft()
        # else:
        #     self._point_idx+=1
        
        for tree in self._forest:
            # If tree is above permitted size...
            tree_count += 1
            if forget_point is not None:
                tree.forget_point(index=forget_point)

            try:
                tree.insert_point(self._shingle_points, index=self._point_idx, tolerance=0.01)
                point_codisp = tree.codisp(self._point_idx)
            except AssertionError as err:
                print(f'{err}, assertion error')
                point_codisp = 0

            avg_codisp += point_codisp / tree_count
            codisp_list.append(point_codisp)

        self._point_idx+=1
        self._point_idx = self._point_idx % self._tree_size

        self._codisp_list.append(avg_codisp)
        if len(self._codisp_list) > 50:
            self._codisp_list.pop(0)

        #qt = np.median(self._codisp_list) + 2*np.std(self._codisp_list)
        qt = np.median(self._codisp_list)

        result_num = 1 - 1 / (1 + np.log(1 + avg_codisp))
        return (result_num,)
        #return (int(avg_codisp > qt),)
