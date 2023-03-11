import rrcf
import numpy as np

if __name__ == '__main__':
    index_arr = np.arange(256)
    points = rrcf.shingle(index_arr, size=4)
    for i in range(450):
        print(index_arr[0])
        index_arr = np.roll(index_arr,-1)
    #print(index_arr)
