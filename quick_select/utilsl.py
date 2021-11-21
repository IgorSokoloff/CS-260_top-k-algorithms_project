from random import randint
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(15000)

N_COMPARISONS_QS__ = 0

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def key(x):
    return -x[1], x[2], x[0]


# Partition using Lomuto partition scheme
def partition(arr, left, right, p, key):
    pivot = arr[p]
    swap(arr, p, right)
    p = left
    for i in range(left, right):
        quickselect.N_COMPARISONS_QS += 1
        if key(arr[i]) <= key(pivot):
            swap(arr, i, p)
            p = p + 1
    swap(arr, p, right)
    return p


def quickselect(arr, k, left=None, right=None, key=lambda x: x):
    if left is None or right is None:
        quickselect.N_COMPARISONS_QS = 0
        arr = arr.copy()
        left = 0
        right = len(arr) - 1
    if left == right:
        return arr[left]
    p = randint(left, right)
    p = partition(arr, left, right, p, key)
    quickselect.N_COMPARISONS_QS += 1
    if k == p:
        return arr[k]
    elif k < p:
        return quickselect(arr, k, left, p - 1, key)
    else:
        return quickselect(arr, k, p + 1, right, key)


def topk(arr, k, key=lambda x: x, inplace=True):
    # key=lambda x: -key(x) we find top, so the order is reversed
    n = len(arr)
    if not inplace:
        arr = arr.copy()
    if k < 1 or k > n:
        raise UserWarning('k Value is outside of 1...N')
    kth = quickselect(arr, k-1, key=lambda x: -key(x))
    equals = []
    removed = 0
    for i in range(len(arr)):
        quickselect.N_COMPARISONS_QS += 1
        if arr[i] < kth:
            removed += 1
            arr[i] = 0
        elif kth == arr[i]:
            equals.append(i)
    #stupid part
    for i in equals:
        quickselect.N_COMPARISONS_QS += 1
        if n - removed > k:
            arr[i] = 0
            removed += 1
    if not inplace:
        return arr

def test():
    arr = [1, 9, 4, 5, 63]
    top = topk(arr, 1, inplace=False)
    assert top == [0, 0, 0, 0, 63]
    top = topk(arr, 2, inplace=False)
    assert top == [0, 9, 0, 0, 63]
    top = topk(arr, 3, inplace=False)
    assert top == [0, 9, 0, 5, 63]
    top = topk(arr, 4, inplace=False)
    assert top == [0, 9, 4, 5, 63]
    top = topk(arr, 5, inplace=False)
    assert top == [1, 9, 4, 5, 63]

def test_complexity(type='average', nmax = 5000, nmin=20, step = 1000, average = 10, k=10, output=None):
    ns = np.arange(nmin,nmax,step)
    niters = np.zeros(len(ns))
    iters = 0
    for i, n in enumerate(ns):
        if type == 'worst':
            arr = np.arange(n, dtype=np.int)[::-1]
        else:
            arr = np.random.rand(n)
        for _ in range(average):
            topk(arr,k)
            iters += quickselect.N_COMPARISONS_QS
        iters /= average
        niters[i] = iters
    if output is None:
        plt.plot(ns,niters)
        plt.show()


if __name__ == '__main__':
    arr = [1, 1, 1, 1, 1, 1, 1, 1, 1] * 1000
    test_complexity(type='worst')
    exit(0)
    top = quickselect(arr, 2)
    topk(arr, 7)
    print(quickselect.N_COMPARISONS_QS)
    test()
    print(quickselect.N_COMPARISONS_QS)