from random import randint
import numpy as np
import matplotlib.pyplot as plt
import sys
from enum import Enum
import math
sys.setrecursionlimit(15000)

class PivotType(Enum):
        RANDOM = 1
        DETERMINISTIC = 2
        MEDIAN = 3
class Quickselect:
    def __init__(self):
        self.numberOfComparisons = 0


    def swap(self, arr, i, j):
        arr[i], arr[j] = arr[j], arr[i]


    def key(self, x):
        return -x[1], x[2], x[0]


    # Partition using Lomuto partition scheme
    def partition(self, arr, left, right, p, key):
        pivot = arr[p]
        self.swap(arr, p, right)
        p = left
        for i in range(left, right):
            self.numberOfComparisons += 1
            if key(abs(arr[i])) <= key(abs(pivot)):
                self.swap(arr, i, p)
                p = p + 1
        self.swap(arr, p, right)
        return p

    # Source: Cormen, Thomas H.; Leiserson, Charles E.; Rivest, Ronald L.; Stein, Clifford (2009) [1990]. Introduction to Algorithms (3rd ed.). MIT Press and McGraw-Hill. ISBN 0-262-03384-4.
    # Return median of a group with at most 5 elements 
    def partition5(self, arr, left, right):
        i = left + 1
        while i <= right:
            j = i
            while j > left and arr[j-1] > arr[j]:
                self.numberOfComparisons += 1
                self.swap(arr, j-1, j)
                j = j - 1
            i =  i + 1
                
        return math.floor((left + right) / 2)


    # Source: Cormen, Thomas H.; Leiserson, Charles E.; Rivest, Ronald L.; Stein, Clifford (2009) [1990]. Introduction to Algorithms (3rd ed.). MIT Press and McGraw-Hill. ISBN 0-262-03384-4.
    # Median-of-medians algorithm
    # 1- Divides the array into groups of 5 elements
    # 2- Get the median of each group and move it to the front. By the end, all medians will be in the first n/5 positions
    # 3- Compute the true median of the n/5 medians
    # 4- Call quickselect 
    def medianPivot(self, arr, left, right):
        # for 5 or less elements just get median
        if (right - left) < 5:
            return self.partition5(arr, left, right)
        # Otherwise move the medians of five-element subgroups to the first n/5 positions
        for i in range(left, right, 5):
            # Get the median position of the i'th five-element subgroup
            subRight = i + 4
            self.numberOfComparisons += 1
            if subRight > right:
                subRight = right
            median5 = self.partition5(arr, i, subRight)
            self.swap(arr, median5, left + math.floor((i - left)/5))

        # compute the median of the n/5 medians-of-five
        mid = (right - left) // 10 + left + 1
        return self.quickselect(arr, mid, left=left, right=left + math.floor((right - left) / 5), pivotType=PivotType.MEDIAN, returnIndex=True)


    def quickselect(self, arr, k, left, right, key=lambda x: x, pivotType=PivotType.RANDOM, returnIndex = False):
        # if left is None or right is None:
        #     self.numberOfComparisons = 0  
        #     arr = arr.copy()
        #     left = 0
        #     right = len(arr) - 1

        if left == right:
            if returnIndex:
                return left
            else:
                return arr[left]
        self.numberOfComparisons += 1

        # Random Pivot
        if pivotType == PivotType.RANDOM:
            p = randint(left, right)
            
        # Deterministic Pivot - choose middle index
        elif pivotType == PivotType.DETERMINISTIC:
            p = left + (abs(left-right))//2

        # Median-of-medians Pivot 
        else:
            p = self.medianPivot(arr, left, right)

        p = self.partition(arr, left, right, p, key)

        self.numberOfComparisons += 1
        if k == p:
            if returnIndex:
                return k
            else:
                return arr[k]
        elif k < p:
            self.numberOfComparisons += 1
            return self.quickselect(arr, k, left, p - 1, key, pivotType, returnIndex=returnIndex)
        else:
            return self.quickselect(arr, k, p + 1, right, key, pivotType, returnIndex=returnIndex)


    def getTopK(self, arr, k, key=lambda x: x, inplace=True, pivotType = PivotType.RANDOM):
        self.numberOfComparisons = 0
        # key=lambda x: -key(x) we find top, so the order is reversed
        n = len(arr)
        if not inplace:
            arr = arr.copy()
        if k < 1 or k > n:
            raise UserWarning('k Value is outside of 1...N')
        
        kth = self.quickselect(arr.copy(), k-1, left = 0, right = len(arr) - 1, key=lambda x: -key(x), pivotType=pivotType)
        equals = []
        removed = 0
        for i in range(len(arr)):
            self.numberOfComparisons += 1
            if abs(arr[i]) < abs(kth):
                removed += 1
                arr[i] = 0
            elif abs(kth) == abs(arr[i]):
                self.numberOfComparisons += 1
                equals.append(i)
        #stupid part
        for i in equals:
            self.numberOfComparisons += 1
            if n - removed > k:
                arr[i] = 0
                removed += 1
        if not inplace:
            return arr, self.numberOfComparisons

    def test(self, pivotType=PivotType.RANDOM):
        arr = [1, 9, 4, 5, 63]
        top = self.getTopK(arr, 1, inplace=False, pivotType=pivotType)
        assert top == [0, 0, 0, 0, 63]
        top = self.getTopK(arr, 2, inplace=False, pivotType=pivotType)
        assert top == [0, 9, 0, 0, 63]
        top = self.getTopK(arr, 3, inplace=False, pivotType=pivotType)
        assert top == [0, 9, 0, 5, 63]
        top = self.getTopK(arr, 4, inplace=False, pivotType=pivotType)
        assert top == [0, 9, 4, 5, 63]
        top = self.getTopK(arr, 5, inplace=False, pivotType=pivotType)
        assert top == [1, 9, 4, 5, 63]

    def test_complexity(self, type='average', nmax = 5000, nmin=20, step = 1000, average = 10, k=10, output=None, pivotType=PivotType.RANDOM):
        ns = np.arange(nmin,nmax,step)
        niters = np.zeros(len(ns))
        iters = 0
        print(ns)
        for i, n in enumerate(ns):
            if type == 'worst':
                arr = np.arange(n, dtype=np.int)[::-1]
            else:
                arr = np.random.rand(n)
            for _ in range(average):
                self.getTopK(arr,k,pivotType=pivotType)
                iters += self.numberOfComparisons
            iters /= average
            niters[i] = iters
        if output is None:
            plt.plot(ns,niters)
            plt.show()



#TEST 1
# myQuickselect = Quickselect()
# arr = [1, 1, 1, 1, 1, 1, 1, 1, 1] * 10
# quickselect.test_complexity(type='worst')
# exit(0)
# pivotType = PivotType.DETERMINISTIC
# top = myQuickselect.quickselect(arr, 2, left = 0, right = len(arr) - 1, pivotType=pivotType)
# myQuickselect.getTopK(arr, 7, pivotType=pivotType)
# print(myQuickselect.numberOfComparisons)
# myQuickselect.test(pivotType=pivotType)
# print(myQuickselect.numberOfComparisons)



#TEST 2
# myQuickselect = Quickselect()    
# arr = np.random.rand(100000)
# top, numberOfComparisons = myQuickselect.getTopK(arr, 5, inplace=False, pivotType=PivotType.RANDOM)
# print("Random Pivot:\t\t", numberOfComparisons)

# top, numberOfComparisons = myQuickselect.getTopK(arr, 5, inplace=False, pivotType=PivotType.DETERMINISTIC)
# print("Deterministic Pivot:\t", numberOfComparisons)

# top, numberOfComparisons = myQuickselect.getTopK(arr, 5, inplace=False, pivotType=PivotType.MEDIAN)
# print("Median Pivot:\t\t", numberOfComparisons)