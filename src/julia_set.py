from typing import List
import time
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.core.fromnumeric import mean

x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8 
# These parameter will affect plot a lot!
# c_real, c_imag = -0.82772, -.82193
c_real, c_imag = -0.62772, -.42193

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2-t1:.4f} seconds")
        return result

    return measure_time


def plot(out: List[int]):
    mat = list_to_matrix(out)
    x, y = range(0, mat.shape[0]), range(0, mat.shape[1])
    fig, ax = plt.subplots()
    ax.imshow(mat, interpolation='nearest', cmap=cm.hot)
    plt.show()


def list_to_matrix(output):
    length = len(output)
    n = int(length ** 0.5)
    matrix = np.array(output).reshape((n, n))
    return matrix

# @timefn
def calculate_z_serial_purepython(max_iter, zs, cs):
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while (abs(z)) < 2 and n < max_iter:
            z = z*z + c
            n += 1
        output[i] = n
    return output


def calc_pure_python(width: int, max_iter: int):
    """ Create a list of complex coordinates """
    x_step = ( x2 - x1 ) / width
    y_step = ( y2 - y1 ) / width
    x, y = [], []
    ycood = y1 
    while ycood < y2:
        y.append(ycood)
        ycood += y_step
    xcood = x1
    while xcood < x2:
        x.append(xcood)
        xcood += x_step

    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord)) 
            cs.append(complex(c_real, c_imag))
    
    print(f"Length of x: {len(x)}")
    print(f"Total elements: {len(zs)}")

    start_time = time.time()
    output = calculate_z_serial_purepython(max_iter, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(f"{calculate_z_serial_purepython.__name__} took {secs} seconds.")
    # assert sum(output) == 33219980
    print(f"{sum(output) = }")
    return output


if __name__ == "__main__":

    width = 1000
    max_iter = 300
    out = calc_pure_python(width, max_iter)

    # plot result
    # plot(out)