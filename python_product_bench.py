import timeit


setup = '''
import numpy as np

size = {}
new_axes = int(size/2)
matrix1 = np.random.random_sample(tuple([2]*size +  [1]*new_axes))
matrix2 = np.random.random_sample(tuple([1]*new_axes + [2]*size))
'''

for size in range(2, 13, 2):
    time_taken = timeit.Timer('matrix1*matrix2', setup=setup.format(size)).repeat(10, 10000)
    per_loop = min(time_taken) / 10000
    print('Benchmark ' + str(size) + ": " + str(per_loop*1000000) + " us")