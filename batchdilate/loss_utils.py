from numba import njit
import numpy as np

@njit
def np_apply_along_axis(func1d, axis, arr):
    """ credits to @joelrich : https://github.com/joelrich """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.zeros(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit
def np_max_along_axis(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@njit
def np_sum_along_axis(array, axis):
    return np_apply_along_axis(np.sum, axis, array)


@njit
def my_max2_njit(x, gamma):
    max_x = np_max_along_axis(x, 1).reshape(-1, 1)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np_sum_along_axis(exp_x, 1).reshape(-1, 1)
    return gamma * np.log(Z.reshape(-1)) + max_x.reshape(-1), exp_x / Z


@njit
def my_min2(x, gamma):
    min_x, argmax_x = my_max2_njit(-x, gamma)
    return - min_x, argmax_x


@njit
def my_max_hessian_product2(p, z, gamma):
    b, c, _ = p.shape
    interm_sum = np_sum_along_axis((p * z).reshape(-1, 3), 1)
    sum_recalculated = p * interm_sum.reshape(b, c, 1)
    return (p * z - sum_recalculated) / gamma


@njit
def my_min_hessian_product2(p, z, gamma):
    return - my_max_hessian_product2(p, z, gamma)
