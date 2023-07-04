"""
A script to generate a dataset matrix from a given image.
The dataset is used to train an n*2*n encoder-decoder model.
Use the first neuron in the hidden layer as x and the second neuron as y should be able to reconstruct the image.

Author: ROYFXY (https://github.com/royfxy)
Version: 1.0
Date: 2023-07-05
"""

import numpy as np
import torch

"""
========================================
        Change the image here
========================================
"""
IMAGE = """*------*------*
           ----------***--
           ----*-----*--*-
           ---*-*---*--*--
           --*---***--*---
           -*----------*--
           --*---------*--
           ---****-----*--
           -------*****---
           *-------------*"""


def parse(img):
    # filter out characters that are not -, * or \n
    img = "".join(c for c in img if c in "-*\n")
    # split into lines
    img = img.split("\n")
    # assert all lines have the same length
    assert len(set(map(len, img))) == 1, "All lines must have the same length"
    # assert that all corners are stars
    assert (
        img[0][0] == img[0][-1] == img[-1][0] == img[-1][-1] == "*"
    ), "All corners must be stars"
    # assert that all columns and rows must have at least one star
    assert all(["*" in row for row in img]) and all(
        ["*" in col for col in np.transpose(list(map(lambda row: [*row], img)))]
    ), "All columns and rows must have at least one star"
    # get the coordinates of the stars
    stars_coords = [
        (i, j) for i, line in enumerate(img) for j, c in enumerate(line) if c == "*"
    ]
    return stars_coords


def gen_matrix(coords):
    num_row = len(coords)
    # initialize matrix
    matrix = []
    # fill in matrix
    vec_row = np.zeros(num_row).tolist()
    coords.sort(key=lambda x: x[1])
    for i, coord in enumerate(coords):
        if coord[1] != 0 and coord[1] != coords[i - 1][1]:
            matrix.append(vec_row)
            vec_row = vec_row.copy()
        vec_row[i] = coord
    vec_col = np.zeros(num_row).tolist()
    coords.sort(key=lambda x: x[0])
    for i, coord in enumerate(coords):
        if coord[0] != 0 and coord[0] != coords[i - 1][0]:
            matrix.append(vec_col)
            vec_col = vec_col.copy()
        vec_col[vec_row.index(coord)] = coord

    # make matrix zero and one
    matrix = np.array(matrix, dtype=object)
    matrix[matrix != 0] = 1
    matrix = matrix.astype(np.float32)
    return matrix.transpose()


def check_row_duplicates(matrix):
    duplicate_rows = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            if np.array_equal(matrix[i], matrix[j]):
                print("Found duplicate rows: ", i, j)
                duplicate_rows.append(i)
    return np.unique(duplicate_rows)


if __name__ == "__main__":
    matrix = gen_matrix(parse(IMAGE))
    check_row_duplicates(matrix)
    matrix = torch.from_numpy(matrix).int()
    print(matrix.shape)
    print(matrix)
