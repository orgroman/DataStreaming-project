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
        self._shingle_points = []
        self._timestamp_idx = []

        # Create a forest of empty trees
        self._forest = []
        for _ in range(self._num_trees):
            tree = rrcf.RCTree()
            self._forest.append(tree)

    def handleRecord(self, inputData, label):
        # For each tree in the forest...
        avg_codisp = 0
        timestamp = inputData['timestamp']
        point = inputData['value']
        self._shingle_points.append(point)
        if len(self._shingle_points) > self._shingle_size:
            self._shingle_points.pop(0)
        else:
            return (0,)

        tree_count = 0
        min_codisp = 1e9
        codisp_list = []

        self._timestamp_idx.append(timestamp)

        forget_point = None
        if len(self._timestamp_idx) > self._tree_size:
            forget_point = self._timestamp_idx.pop(0)

        for tree in self._forest:
            # If tree is above permitted size...
            tree_count += 1
            #point_idx = len(tree.leaves)
            if forget_point:
                tree.forget_point(index=forget_point)

            # if len(tree.leaves) >= self._tree_size:
            #     # # Insert the new point into the tree
            #     #point_idx = int(random.random() * self._tree_size)
            #     tree.forget_point(index=point_idx)
            tree.insert_point(self._shingle_points, index=timestamp, tolerance=0.01)
            point_codisp = tree.codisp(timestamp)
            avg_codisp += point_codisp / tree_count
            # min_codisp = min(min_codisp, point_codisp)
            codisp_list.append(point_codisp)

        #print(f'avg: {avg_codisp}')
        median_codisp = np.median(codisp_list)

        codisp_std = np.std(codisp_list)
        codisp_mean = np.mean(codisp_list)

        # if codisp_std != 0:
        #     #zscore_codisp = (np.array(codisp_list)-codisp_mean)/codisp_std
        #     #sig_array = sigmoid_array(zscore_codisp)
        #     #zmean = np.quantile(zscore_codisp, 0.9)
        #     #zmean = np.max(zscore_codisp)
        #     #sgn = 'positive' if zmean > 1 else 'negative'
        #     #sgn = sigmoid(zmean)
        #     sgn = 1/(1+avg_codisp)
        #     print(f'label:{label},sig:{sgn}')
        # else:
        #     sig_array = np.zeros_like(codisp_list)


        # if label > 0:
        #     print(f'label:{label},sig:{np.median(sig_array)}')

        #print(f'max={sig_array.max()}, median={np.median(sig_array)}, mean={np.mean(sig_array)}, min={sig_array.min()}')

        # self._point_idx += 1
        # self._point_idx = self._point_idx % self._tree_size

        self._codisp_list.append(avg_codisp)
        if len(self._codisp_list) > 50:
            self._codisp_list.pop(0)


        #
        qt = np.median(self._codisp_list) + 2*np.std(self._codisp_list)
        # if avg_codisp > qt:
        #     print(f'qt = {qt}, avg = {avg_codisp}, median={median_codisp}, min={min_codisp}')

        #qt = np.quantile(self._codisp_list, 0.999)
        # if avg_codisp < qt:
        #     avg_codisp = 0

        # self._codisp_threshold = qt

        # result = np.quantile(self._codisp_list, 0.99)
        #
        result_num = 1 - 1 / (1 + np.log(1 + avg_codisp))
        #result_num = 1-1/(1+median_codisp)
        # mean_sig_array = np.mean(sig_array)
        # median_sig_array = np.median(sig_array)
        # max_sig_array = np.max(sig_array)
        #
        # if max_sig_array > 0.9 and mean_sig_array > 0.5:
        #     print(f'max={sig_array.max()}, median={median_sig_array}, mean={np.mean(sig_array)}, min={sig_array.min()}')

        # if result_num > 0.8:
        #     print(f'Anomaly! qt = {qt}, avg = {avg_codisp}, median={median_codisp}, min={min_codisp}')

        # result = result_num if min_codisp > qt else 0
        # if result > 0:
        #     print(f'Anomaly! - {result}, qt = {qt}, avg = {avg_codisp}, median={median_codisp}, min={min_codisp}')

        # result = avg_codisp > self._codisp_threshold

        #print(median_codisp)
        # print(self._codisp_list)
        # print(f'label: {label}, qt: {avg_codisp > qt}')
        return (int(avg_codisp > qt),)
