from nab.detectors.base import AnomalyDetector
import random
import numpy as np
import rrcf

class RrctDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        # super(RRCFDetector, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self._num_trees = {}.get('num_trees', 40)
        self._shingle_size = {}.get('shingle_size', 6)

        # Use reservoir sampling to drop or insert points
        self._tree_size = {}.get('tree_size', 256)

        # Create a forest of empty trees
        self._forest = []
        for _ in range(self._num_trees):
            tree = rrcf.RCTree()
            self._forest.append(tree)

    def handleRecord(self, inputData):
        # For each tree in the forest...
        avg_codisp = 0

        for tree in self._forest:
            # If tree is above permitted size...
            if len(tree.leaves) < self._tree_size:
                # Insert the new point into the tree
                point_index = len(tree.leaves)
                tree.insert_point(point, index=point_index)
            else:
                point_index = int(random.random() * self._tree_size)

                # Drop the sampled point for that index and insert a new sample
                tree.forget_point(point_index)

                # Insert the new point into the tree
                tree.insert_point(point, index=point_index)

            new_codisp = tree.codisp(point_index)
            avg_codisp += new_codisp / self._num_trees

        return [avg_codisp]
