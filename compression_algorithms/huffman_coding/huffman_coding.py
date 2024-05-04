# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:23:25 2024

@author: Sai Gunaranjan Pelluri
"""

"""

Huffman coding compression algorithm

In this script, I have implemented a lossless compression coding scheme named Huffman encoding. This algorithm
will stress test our understanding of programming concepts like classes, nodes, linked lists, trees,
tree traversal, recursion, reading/writing to/from a binary file, etc. To test this algorithm,
I have a taken a sample text file named 'uncompressed.txt' of size 700 KB. On running the Huffman coding
compression algorithm on this text file data, I was able to compress the file to 385 KB binary file
named 'compressed.bin'. Then I decompress this compressed binary file to retrieve back the original data.
This decompressed file is named 'decompressed.txt'. I then check the original 'uncompressed.txt' file and
the 'decompressed.txt' file and they match extactly proving that the Huffman coding is a lossless compression
scheme.
I also print the entopy of the text data and also the average code word length (number of bits per symbol).
Entropy is computed as -sum p log2(p) whereas average code word length/expected code word length is computed as
sum codeWordLength * Number of occurences of te CodeWordLength. Theoreticaly, both these measures are same and
that is what is observed as well.


A simple and beautiful explanation and reference implementation for Huffman coding is available in the below links:
    1. https://www.youtube.com/watch?v=_Kl3TtBXxq8
    2. https://www.youtube.com/watch?v=JCOph23TQTY&t=1121s

My implementation is quite different from the above links and is not as optimal as shown in the above tutorials.
But the output is the same.

Action Item:
    1. Extend the scheme to any generic symbols ad not necessarily numbers
    Only leaf nodes can have symbols, rest all nodes have only data [Done]
    2. Show size of the Huffman LUT

"""

import numpy as np



class Node:

    def __init__(self,data, symbol):

        self.data = data
        self.left_child = None
        self.right_child = None
        # self.left_edge = None
        # self.right_edge = None
        self.symbol = symbol


class HuffmanTree:


    def __init__(self,symbolFreqDict):

        self.codeWordDict = {}
        self.listNodes = []
        self.symbolFreqDict = symbolFreqDict
        for key, value in self.symbolFreqDict.items():
            self.listNodes.append(Node(value, key))


    # def sort_nodes(self,listNodes):

    #     n = len(listNodes)
    #     for i in range(n):
    #         for j in range(0, n-i-1):
    #             if listNodes[j].data > listNodes[j+1].data:
    #                 listNodes[j], listNodes[j+1] = listNodes[j+1], listNodes[j]

    #     return listNodes

    def sort_nodes(self,listNodes):

        n = len(listNodes)
        dataList = []
        for i in range(n):
            dataList.append(listNodes[i].data)

        ind = sorted(range(n), key = dataList.__getitem__)
        listNodes = [listNodes[i] for i in ind]

        return listNodes


    def build_huffman_tree(self):

        numElements = len(self.listNodes)
        newNodeList = self.listNodes

        if numElements == 1:
            print('Cannot build a Huffman Tree with 1 element!')
            return

        while numElements > 1:
            newNodeListSorted = self.sort_nodes(newNodeList)
            remList = newNodeListSorted[2::]
            newNodeData = newNodeListSorted[0].data + newNodeListSorted[1].data
            newNode = Node(newNodeData,None)
            newNode.left_child = newNodeListSorted[0]
            newNode.right_child = newNodeListSorted[1]
            # newNode.left_edge = 0
            # newNode.right_edge = 1
            newNodeList = []
            newNodeList.append(newNode)
            newNodeList.extend(remList)
            numElements = len(newNodeList)

        self.HuffmanTreeList = newNodeList



    def generate_huffman_codes(self, rootNode, codeWord):

        if rootNode is None:
            return

        if rootNode.symbol is not None:
            self.codeWordDict[rootNode.symbol] = codeWord

        self.generate_huffman_codes(rootNode.left_child, codeWord + '0')
        self.generate_huffman_codes(rootNode.right_child, codeWord + '1')



    def compute_entropy(self):

        count = 0
        for _, value in self.symbolFreqDict.items():
            count += value

        self.entropy = 0
        for _, value in self.symbolFreqDict.items():
            p = value/count
            self.entropy += (p * np.log2(1/p)) # Entropy = sum (p*log2(1/p))

        self.averageWordLength = 0
        for key, value in self.symbolFreqDict.items():
            codeWordLen = len(self.codeWordDict[key])
            probOccurence = self.symbolFreqDict[key] / count

            self.averageWordLength += codeWordLen*probOccurence # E[x] = sum (x_i) * P(x=x_i)

        print('Entropy = {0:.2f} bits'.format(self.entropy))
        print('Average code length = {0:.2f}'.format(self.averageWordLength))


    def encode_data(self,inputTextFileName, outputCompressBinFileName):


        with open(inputTextFileName, "r") as f:
            binaryString = ""
            while True:
                symbol = f.read(1)
                if not symbol:
                    break
                binaryString += self.codeWordDict[symbol]

        residualBits = np.mod(len(binaryString),8)

        print('\nActual data size = {} bytes'.format(len(binaryString)/8))
        print('Residual bits after bytes formation = {}'.format(residualBits))

        numAppendedBits = 8 - residualBits
        numAppendedBitsBinary = bin(numAppendedBits)[2::]
        """ Header byte carries the info about how many bits were appended at the end. To make the header
        also as a byte, we appened 8-len(numAppendedBitsBinary) number of 0 bits to the binary representation of
        # of appened bits to make the header 1 byte"""
        headerByte = '0'*(8-len(numAppendedBitsBinary))  + numAppendedBitsBinary

        binaryString = headerByte + binaryString + '0'*numAppendedBits

        print('Data size post appending = {} bytes'.format(len(binaryString)/8))

        f = open(outputCompressBinFileName, 'wb')
        chrString = bytearray(int(binaryString[x:x+8], 2) for x in range(0, len(binaryString), 8))
        f.write(chrString)
        f.close()

        print('\nCompleted data encoding....')



    def decode_data_wrapper(self, CompressBinFileName, outputDecompressedTextFile):


        self.decodedbinaryString = "".join(f"{n:08b}" for n in open(CompressBinFileName, "rb").read()) # Converts byte array to bitstring
        headerByteString = self.decodedbinaryString[0:8] # Extract the header byte carrying info about the number of 0 bits padded at end
        numBitsappended = int(headerByteString,2)
        self.decodedbinaryString = self.decodedbinaryString[8:-numBitsappended] # Strip the header and the extra padded 0 bits

        self.symbolList = [] # create a placeholder to decoded symbols

        print('\nCommencing decoding....')

        count = 0
        n = 0
        self.decode_data(count,n)
        # self.decode_data_recursion(count,n) # Recursion based decoding works only for smaller lenghts of texts. If the text size is large, then recursion does not work and throws an error

        textString = "".join(self.symbolList) # Collapse the list of symbols(strings in this case) to a large string/text

        f = open(outputDecompressedTextFile, 'w')
        f.write(textString)
        f.close()

        print('\nDecoding completed!')




    # def decode_data_recursion(self,count,n):

    #     if count == len(self.decodedbinaryString): # Replace later with count == len(self.decodedbinaryString)
    #         return
    #     if self.decodedbinaryString[count:count+n] in self.codeWordDict.values():
    #         for symbol, codeword in self.codeWordDict.items():
    #             if (codeword == self.decodedbinaryString[count:count+n]):
    #                 self.symbolList.append(symbol)
    #                 break


    #         count = count + n #+ 1
    #         n = 0

    #     self.count = count

    #     self.decode_data_recursion(count,n+1)


    def decode_data(self,count,n):


        while count < len(self.decodedbinaryString):
            while True:
                if count+n > len(self.decodedbinaryString):
                    return
                if self.decodedbinaryString[count:count+n] in self.codeWordDict.values():
                    for symbol, codeword in self.codeWordDict.items():
                        if (codeword == self.decodedbinaryString[count:count+n]):
                            self.symbolList.append(symbol)
                            break

                    count = count + n
                    n = 0
                    break
                else:
                    n += 1

















