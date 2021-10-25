from random import randint


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
        if key(arr[i]) <= key(pivot):
            swap(arr, i, p)
            p = p + 1
    swap(arr, p, right)
    return p


def quickselect(arr, k, left=None, right=None, key=lambda x: x):
    if left is None or right is None:
        arr = arr.copy()
        left = 0
        right = len(arr) - 1
    if left == right:
        return arr[left]
    p = randint(left, right)
    p = partition(arr, left, right, p, key)
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
        if arr[i] < kth:
            removed += 1
            arr[i] = 0
        elif kth == arr[i]:
            equals.append(i)
    #stupid part
    for i in equals:
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



if __name__ == '__main__':
    arr = [1, 3, 20, 30, 9, 2, 31, 7, 2]
    top = quickselect(arr, 2)
    topk(arr, 7)
    test()