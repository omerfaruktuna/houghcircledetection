'''
PREPROCESSING STEPS

Apply filtering operation to detect edges

PSEUDOCODE for HOUGH CIRCLES METHOD

Input a binary edge image E(x,y)
Initialize accumulator array A(x0,y0,r) to zeros
Dimensions of A :
x0 --> rows of E matrix
y0 --> columns of E matrix
r  --> sqrt( power(rows of E matrix,2) + power(columns of E matrix,2) )


for all x:
    for all y:
        if E(x,y):
            for all x0:
                for all y0:
                    r = sqrt( power((x-x0),2) + power((y-y0),2) )
                    increment A at (x0,y0,r)
                    end
                end
            end
        end
    end
end

Search for the peaks in A(x0,y0,r) - the corresponding indices of A are the parameter of the detected circles.

When we inspect the input image, we can see that the average radius length of the circles is around 10 pixel.

So in the accumulation matrix, we only consider the possible circles coordinates (x0,y0) with r = 10

The resulting accumulator can be used to detect the number of circles.

I have coded a simple function find_number_of_circles(Accumulator_Matrix, ratioparameter) which takes argument as Accumulator matrix and a threshold and outputs the # of circles.

'''

import numpy as np
from PIL import Image
import cv2
import math


def zero_pad(X, pad):

    X_pad = np.pad(X, ((pad, pad), (pad, pad)), 'constant', constant_values=0)

    return X_pad

def conv_single_step(img_slice, W):

    s = np.multiply(img_slice, W)
    Z = np.sum(s)

    return Z

def conv_operation_gradient(image, Wx, Wy):

    (n_H_prev, n_W_prev) = image.shape
    (f, f) = Wx.shape

    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1

    Z = np.zeros((n_H, n_W))

    for h in range(n_H):
        for w in range(n_W):
            vert_start = h
            vert_end = vert_start + f
            horiz_start = w
            horiz_end = horiz_start + f

            img_slice = image[vert_start:vert_end, horiz_start:horiz_end]
            calculated_calue_x = int(conv_single_step(img_slice, Wx[:,:]))
            calculated_calue_y = int(conv_single_step(img_slice, Wy[:, :]))
            Z[h, w] = int(math.sqrt((calculated_calue_x**2) + (calculated_calue_y**2)))


    return Z

def find_accumulator_matrix(input_image, searched_radius):

    input_image_pad = zero_pad(input_image, 15)

    row, column = input_image_pad.shape

    R = int(math.sqrt(row ** 2 + column ** 2))

    Accumulator = np.zeros((row, column, R))

    for i in range(row):
        for j in range(column):
            if input_image_pad[i][j] != 0:
                for m in range(i-15,i+15):
                    for n in range(j-15,j+15):
                        Radius = int(math.sqrt(((i-m) ** 2) + ((j-n) ** 2)))
                        Accumulator[m][n][Radius] += 1


    New_Accumulator = Accumulator[:,:,searched_radius]
    return New_Accumulator

def find_number_of_circles(Accumulator_Matrix, ratioparameter):

    max_value = np.max(Accumulator_Matrix)

    indices = np.argwhere(Accumulator_Matrix > (max_value * ratioparameter))

    x_cord = []
    y_cord = []


    for ind in indices:
        x_cord.append(ind[0])
        y_cord.append(ind[1])

    count = 0

    for i in range(len(x_cord)):

        if i == 0:
            count += 1
            print(i)
            continue

        if abs(x_cord[i] - x_cord[i-1]) < 3 and abs(y_cord[i] - y_cord[i-1]) < 3:
            continue

        if (x_cord[i]-1) in x_cord:

            index_list = [index for index, value in enumerate(x_cord) if value == (x_cord[i]-1)]

            for a in index_list:
                if abs(y_cord[a] - y_cord[i]) > 3:
                    break
            continue


        if (x_cord[i]-2) in x_cord:

            index_list = [index for index, value in enumerate(x_cord) if value == (x_cord[i]-2)]

            for a in index_list:
                if abs(y_cord[a] - y_cord[i]) > 3:
                    break
            continue

        count += 1

    print("Number of circles in input image is {}".format(count))


ratio = 0.7
Target_Radius = 10

pic = cv2.imread('binary_circles.tif', 0)

img = np.array(pic)


# Kernel for Gradient in x-direction
Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# Kernel for Gradient in y-direction
Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

img_pad = zero_pad(img, 1)

edges = conv_operation_gradient(img_pad, Kx, Ky)

im_1 = Image.fromarray(edges)
im_1.save("binary_circles_edges.tif")

A = find_accumulator_matrix(edges, Target_Radius)
find_number_of_circles(A, ratio)


