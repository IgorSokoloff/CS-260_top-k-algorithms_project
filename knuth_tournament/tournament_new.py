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

    #    @param inputArray
    #               unordered array of non-negative integers
    #    @param k
    #               order of maximum value desired
    #    @return kth maximum value
    #

    def getKthMaximum(self, inputArray, k):
        return self.findKthMaximum(inputArray, k)[k - 1]

    #   @param inputArray
    #              unordered array of non-negative integers
    #   @param k
    #              ordered number of maximum values
    #   @return k ordered maximum values
    #

    def getTopK(self, inputArray, k):
        self.numberOfComparisons = 0
        outputArray = np.zeros(len(inputArray))
        topKElements, numberOfComparisons = self.findKthMaximum(inputArray, k)
        for element, index in topKElements:
            outputArray[index] = element
        return outputArray, numberOfComparisons

    #   First output tree will be obtained using tournament method. For k
    #   maximum, the output tree will be backtracked k-1 times for each sub tree
    #   identified by the maximum value in the aggregate adjacency list obtained
    #   from each run. The maximum value after each run will be recorded and
    #   successive runs will produce next maximum value.
    #
    #   @param inputArray
    #   @param k
    #   @return ordered array of k maximum elements
    #

    def findKthMaximum(self, inputArray, k):
        inputWithIndex = []
        for i in range(len(inputArray)):
          inputWithIndex.append((inputArray[i], i))


        partiallySorted = [None] * k
        outputTree = self.getOutputTree(inputWithIndex)
        # getOutputTree() run the tournament for all elements with O(d-1).
        # Thus, at this point our self.numberOfComparisons should equal to d-1


        root = self.getRootElement(outputTree)

        partiallySorted[0] = root
        rootIndex = 0
        level = len(outputTree)
        fullAdjacencyList = []
        for i in range(1, k):
            fullAdjacencyList +=self.getAdjacencyList(outputTree, root, level, rootIndex)
            
            # This is expected to take O(log(d)) comparisons
            kThMax, maxIndex = self.getKthMaximumFromAdjacencyList(fullAdjacencyList, i, root)
            
            root = kThMax.element
            partiallySorted[i] = root
            level = kThMax.treeDepth + 1
            rootIndex = kThMax.treeIndex

            # Delete kth element from fullAdjacencyList
            del fullAdjacencyList[maxIndex]
 


        return partiallySorted, self.numberOfComparisons

    #   Takes an input array and generated a two-dimensional array whose rows are
    #   generated by comparing adjacent elements and selecting maximum of two
    #   elements. Thus the output is inverse triangle (root at bottom)
    #
    #   @param values
    #   @return
    #

    def getOutputTree(self, values):
        # if self.debug:
            # print("Tree:")
            # print("========")
        size = len(values)
        treeDepth = math.log(size) / math.log(2)
        intTreeDepth = math.ceil(treeDepth) + 1
        outputTree = [[] for i in range(intTreeDepth)]

        # first row is the input
        outputTree[0] = values
        # if self.debug: print(outputTree[0])

        currentRow = values
        # intnextRow = None
        for i in range(1, intTreeDepth):
            nextRow = self.getNextRow(currentRow)
            outputTree[i] = nextRow
            currentRow = nextRow
            # if self.debug: print(outputTree[i])
        # if self.debug: print("========")

        return outputTree

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

    #  From the passed full adjacency list and max value scans the list and
    #  returns the information about next maximum value. It returns int array
    #  with two values:
    #  first value: index of the back-track (the max value was found in the
    #  Adjacency list for max value, second max etc.)
    #  second value: index within the identified run.
    #
    #  @param fullAdjacencyList
    #             Adjacency list obtained after k-1 backtracks
    #  @param kth
    #             Order of maximum value desired
    #  @param kMinusOneMin
    #             value of k-1 max element
    #  @return
    #
    def getKthMaximumFromAdjacencyList(self, fullAdjacencyList, kth, kMinusOneMin):
        kThMax = self.AdjacencyElement((0, -1), -1, -1)
        temp = None
        maxIndex = 0

        # maxIndex = np.argmax(fullAdjacencyList)
        # kThMax = fullAdjacencyList[maxIndex]
        # self.numberOfComparisons += len(fullAdjacencyList)

        for i in range(0, len(fullAdjacencyList)):
            temp = fullAdjacencyList[i]

            #This condition is useful if we don't want to count duplicates
            #if (temp[0] < kMinusOneMin[0]) and (temp[0] > kThMax[0]):

            self.numberOfComparisons += 1
            if (abs(temp.getElementValue()) > abs(kThMax.getElementValue())):
                kThMax = temp
                maxIndex = i
        return kThMax, maxIndex

    #  Back-tracks a sub-tree (specified by the level and index) parameter and
    #  returns array of elements (during back-track path) along with their index
    #  information. The order elements of output array indicate the level at
    #  which these elements were found (with elements closest to the root at the
    #  end of the list)
    #
    #  Starting from root element (which is maximum element), find the upper of
    #  two adjacent element one row above. One of the two element must be root
    #  element. If the root element is left adjacent, the root index (for one
    #  row above) is two times the root index of any row. For right-adjacent, it
    #  is two times plus one. Select the other element (of two adjacent
    #  elements) as second maximum.
    #
    #  Then move to one row further up and find elements adjacent to lowest
    #  element, again, one of the element must be root element (again, depending
    #  upon the fact that it is left or right adjacent, you can derive the root
    #  index for this row). Compare the other element with the second least
    #  selected in previous step, select the upper of the two and update the
    #  second lowest with this value.
    #
    #  Continue this till you exhaust all the rows of the tree.
    #
    #  @param tree
    #             output tree
    #  @param rootElement
    #             root element (could be of sub-tree or outputtree)
    #  @param level
    #             the level to find the root element. For the output tree the
    #             level is depth of the tree.
    #  @param rootIndex
    #             index for the root element. For output tree it is 0
    #  @return
    #

    def getAdjacencyList(self, tree, rootElement, level, rootIndex):
        adjacencyList = [self.AdjacencyElement(None, None, i) for i in range(level - 1)]
        adjacentleftElement = -1
        adjacentRightElement = -1
        adjacentleftIndex = -1
        adjacentRightIndex = -1
        rowAbove = []


        # we have to scan in reverse order
        for i in reversed(range(1, level)):
            # one row above
            rowAbove = tree[i - 1]
            adjacentleftIndex = rootIndex * 2
            adjacentleftElement = rowAbove[adjacentleftIndex]

            # the root element could be the last element carried from row above
            # because of odd number of elements in array, you need to do
            # following
            # check. if you don't, this case will blow {8, 4, 5, 6, 1, 2}
            self.numberOfComparisons += 1
            if (len(rowAbove) >= ((adjacentleftIndex + 1) + 1)):
                adjacentRightIndex = adjacentleftIndex + 1
                adjacentRightElement = rowAbove[adjacentRightIndex]
            else:
                adjacentRightElement = -1

            # if there is no right adjacent value, then adjacent left must be
            # root continue the loop.
            self.numberOfComparisons += 2
            if adjacentRightElement == -1:
                # just checking for error condition
                if adjacentleftElement != rootElement:
                    raise Exception("This is error condition. Since there "
                                    + " is only one adjacent element (last element), "
                                    + " it must be root element")
                else:
                    rootIndex = rootIndex * 2
                    adjacencyList[i - 1].element = (0, 0)
                    adjacencyList[i - 1].treeIndex = -1
                    continue

            # one of the adjacent number must be root (max value).
            # Get the other number and compared with second max so far
            if adjacentleftElement == rootElement and adjacentRightElement != rootElement:
                self.numberOfComparisons += 2
                rootIndex = rootIndex * 2
                adjacencyList[i - 1].element = adjacentRightElement
                adjacencyList[i - 1].treeIndex = rootIndex + 1
            elif adjacentleftElement != rootElement and adjacentRightElement == rootElement:
                self.numberOfComparisons += 4
                rootIndex = rootIndex * 2 + 1
                adjacencyList[i - 1].element = adjacentleftElement
                adjacencyList[i - 1].treeIndex = rootIndex - 1
            elif adjacentleftElement == rootElement and adjacentRightElement == rootElement:
                self.numberOfComparisons += 6
                # This is case where the root element is repeating, we are not
                # handling this case.
                raise Exception(
                    "Duplicate Elements. This code assumes no repeating elements in the input array")
            else:
                raise Exception("This is error condition. One of the adjacent "
                                + "elements must be root element")

        return adjacencyList

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

    #  Returns the root element of the two-dimensional array.
    #
    #  @param tree
    #  @return
    #

    def getRootElement(self, tree):
        depth = len(tree)
        return tree[depth - 1][0]

    def printRow(self, values):
        for i in range(values):
            print(i + " ")
        print(" ")




# def test():
#     # input = [2, 16, 5, 13, 14, 8, 17, 10]
#     input = np.random.rand(100000)
#     input = np.load("SM_prior-normal_n-100_d-10000.npy")[0]
#     # print(input.shape)
#     tournament = TournamentTopK()
#     k = 6700
#     # k = 3
#     topK, numberOfComparisons = tournament.getTopK(input, k)
#     # print("Top {}: {}".format(k, topK))
#     # print("Total number Of comparisons:", numberOfComparisons)

# import time
# start_time = time.time()
# test()
# end_time = (time.time() - start_time)
# print("%.10f" % end_time)


# input = [2, 16, 5, 12, -14, 8, -17, 10]
# # sample_matrix = np.load("SM_prior-normal_n-100_d-100.npy")
# # input1 = [-0.12113328864268563, 0.3155317518898621, -0.3103550576282231, 0.6748797431357719, -0.07374189293421433]
# # input2 = [-0.04396085650674794, 0.6096451410950671, -0.28666606252832233, 0.3632054776854945, 1.1813390739400884]
# # input = [np.random.normal() for i in list(range(5))]
# # input = sample_matrix[0]
# tournament = TournamentTopK()

# k = 4
# topK, numberOfComparisons = tournament.getTopK(input, k)
# print("Top {}: {}".format(k, topK))
# print("Total number Of comparisons:", numberOfComparisons)
