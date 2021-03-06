# /**
#  * Copyright (c) 2010-2020 Malkit S. Bhasin. All rights reserved.
#  *
#  * All source code and material on this Blog site is the copyright of Malkit S.
#  * Bhasin, 2010 and is protected under copyright laws of the United States. This
#  * source code may not be hosted on any other site without my express, prior,
#  * written permission. Application to host any of the material elsewhere can be
#  * made by contacting me at mbhasin at gmail dot com
#  *
#  * I have made every effort and taken great care in making sure that the source
#  * code and other content included on my web site is technically accurate, but I
#  * disclaim any and all responsibility for any loss, damage or destruction of
#  * data or any other property which may arise from relying on it. I will in no
#  * case be liable for any monetary damages arising from such loss, damage or
#  * destruction.
#  *
#  * I further grant you ("Licensee") a non-exclusive, royalty free, license to
#  * use, modify and redistribute this software in source and binary code form,
#  * provided that i) this copyright notice and license appear on all copies of
#  * the software;
#  *
#  * As with any code, ensure to test this code in a development environment
#  * before attempting to run it in production.
#  *
#  * @author Malkit S. Bhasin
#  *
#  */

import math
import numpy as np
from tqdm import tqdm

class TournamentTopK:

    class AdjacencyElement:
        def __init__(self, element, treeIndex, treeDepth):
            self.element = element # (value, oringial index)
            self.treeIndex = treeIndex
            self.treeDepth = treeDepth
        
        def getElementValue(self):
            return self.element[0]
        
        def __repr__(self):
           return '{0}'.format(self.element)
        
        def __lt__(self, other):
            return ((abs(self.element[0]) < abs(other.element[0])))


    def __init__(self, debug=False):
        self.numberOfComparisons = 0
        self.debug = debug



    def getTopK(self, inputArray, k):
        self.numberOfComparisons = 0
        outputArray = np.zeros(len(inputArray))
        
        if k == 1:
            inp = []
            for i in range(len(inputArray)):
                inp.append((inputArray[i], i))
            tree = self.getOutputTree(inp)
            topk = tree[len(tree)-1][0]
            outputArray[topk[1]] = topk[0]
        else:
            topk = self.getKthElement(inputArray, k)
            for index, element in enumerate(inputArray):
                self.numberOfComparisons += 1 
                if abs(element) >= abs(topk[0]):
                    self.numberOfComparisons += 1 
                    outputArray[index] = element
            
        return outputArray, self.numberOfComparisons

  
    def getKthElement(self, input, k):
        inputCopy = input.copy()
        inp = []
        for i in range(len(inputCopy)):
          inp.append((inputCopy[i], i, i))
        d = len(inp)
        size = d-k+2

        holdout = inp[size:]
        compute = inp[:size]
        tree = self.getOutputTree(compute)
        
        while(True):
            largestElement = tree[len(tree)-1][0]
            treeIndex = largestElement[2]

            self.numberOfComparisons += 1
            if len(holdout) == 0:
                del compute[treeIndex]
                
                tree = self.getOutputTree(compute)
                return tree[len(tree)-1][0]
            else:
                # Take the next item in holdout 
                popedElement = holdout.pop(0)
                compute[treeIndex] = (popedElement[0], popedElement[1], treeIndex)
                self.recomputeTree(tree, treeIndex, compute[treeIndex])

            


    #   Takes an input array and generated a two-dimensional array whose rows are
    #   generated by comparing adjacent elements and selecting maximum of two
    #   elements. Thus the output is inverse triangle (root at bottom)
    #
    #   @param values
    #   @return
    #

    def getOutputTree(self, values):
        size = len(values)
        treeDepth = math.log(size) / math.log(2)
        intTreeDepth = math.ceil(treeDepth) + 1
        outputTree = [[] for i in range(intTreeDepth)]

        # first row is the input
        outputTree[0] = values

        currentRow = values
        for i in range(1, intTreeDepth):
            nextRow = self.getNextRow(currentRow)
            outputTree[i] = nextRow
            currentRow = nextRow

        return outputTree


    def recomputeTree(self, tree, oldElementIndex, newElement):
        intTreeDepth = len(tree)
        index = oldElementIndex

        for i in range(0, intTreeDepth-1): 
            self.numberOfComparisons += 1 
            # Element should be compared with left
            if (index+1) % 2 == 0:
                otherIndex = index-1
            # Element should be compared with right
            else:
                if index == len(tree[i])-1:
                    # No right elements to compare with
                    self.numberOfComparisons += 1 
                    index = max(0, math.floor(index/2))
                    tree[i+1][index] = newElement
                    continue
                otherIndex = index+1


            self.numberOfComparisons += 1 
            if abs(tree[i][index][0]) > abs(tree[i][otherIndex][0]):
                index = max(0, math.floor(index/2))
                tree[i+1][index] = newElement
            else:
                index = max(0, math.floor(index/2))
                tree[i+1][index] = tree[i][otherIndex]                  

        return tree



    #   Compares adjacent elements (starting from index 0), and construct a new
    #   array with elements that are smaller of the adjacent elements.
    #
    #   For even sized input, the resulting array is half the size, for odd size
    #   array, it is half + 1.
    #
    #   @param values
    #   @return
    #

    def getNextRow(self, values):
        rowSize = self.getNextRowSize(values)
        row = [None] * rowSize
        i = 0
        for j in range(0, len(values), 2):

            if j == (len(values) - 1):
                # this is the case where there are odd number of elements
                # in the array. Hence the last loop will have only one element.
                row[i] = values[j]

            else:
                row[i] = self.getMax(values[j], values[j+1])
            i += 1
        self.numberOfComparisons += (i+1) 
        return row


    #  Returns maximum of two passed in values.
    #
    #  @param num1
    #  @param num2
    #  @return
    #
    def getMax(self, num1, num2):
        if abs(num1[0]) > abs(num2[0]):
            return num1
        return num2

    #  following uses Math.ceil(double) to round to upper integer value..since
    #  this function takes double value, diving an int by double results in
    #  double.
    #
    #  Another way of achieving this is for number x divided by n would be -
    #  (x+n-1)/n
    #
    #  @param values
    #  @return
    #
    def getNextRowSize(self, values):
        return math.ceil(len(values) / 2.0)



# import time
# start_time = time.time()
# d = 100000
# input = np.random.rand(d)
# k=int(0.7*d)
# # k = 8
# tournament = TournamentTopK()
# topK, numberOfComparisons = tournament.getTopK(input, k)
# # print(topK)
# print("Total number Of comparisons:", numberOfComparisons)


# end_time = (time.time() - start_time)
# print("%.10f" % end_time)


