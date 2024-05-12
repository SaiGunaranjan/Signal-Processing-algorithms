# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:14:26 2024

@author: Sai Gunaranjan
"""


from collections import Counter
from huffman_coding import HuffmanTree

def load_text_file_character_by_character(filename):
  """Loads a text file character by character.

  Args:
    filename: The name of the text file to load.

  Returns:
    A list of characters in the text file.
  """

  with open(filename, "r") as f:
    characters = []
    while True:
      c = f.read(1)
      if not c:
        break
      characters.append(c)


    return characters

inputTextFile = "uncompressed.txt"
characters = load_text_file_character_by_character(inputTextFile)
occurrences = Counter(characters)
symbolFreqDict = {}
for element, count in occurrences.items():
    symbolFreqDict[element] = count

# symbolFreqDict = {'A':32,'B':16, 'C':8, 'D': 4, 'E':2, 'F': 1} #{'A':2,'B':4, 'C':5, 'D': 9, 'E':19, 'F': 38}

HTree = HuffmanTree(symbolFreqDict)
HTree.build_huffman_tree()
rootNode = HTree.HuffmanTreeList[0]
HTree.generate_huffman_codes(rootNode, "")
# print(HTree.codeWordDict,'\n')
HTree.compute_entropy()

textFileToCompress = inputTextFile
outputCompressedBinaryFile = "compressed.bin"
HTree.encode_data(textFileToCompress, outputCompressedBinaryFile)
outputDecompressedTextFile = "decompressed.txt"
HTree.decode_data_wrapper(outputCompressedBinaryFile, outputDecompressedTextFile)