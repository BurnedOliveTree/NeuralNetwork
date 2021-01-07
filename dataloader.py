import numpy as np
import matplotlib.pyplot as plt
import requests
import gzip


def load_labels(file_name, hmany=None):
    assert("idx1" in file_name)
    result = []
    if ".com" in file_name:
        file = gzip.open(requests.get(file_name, stream=True).raw, 'rb')
    else:
        file = open(file_name, "rb")
    byte = int.from_bytes(file.read(4), byteorder='big')    # magic number
    no = int.from_bytes(file.read(4), byteorder='big')      # number of images
    for img in range(no):
        result.append(int.from_bytes(file.read(1), byteorder='big'))
        if img == hmany-1:
            break
    file.close()
    return np.array(result)


def load_images(file_name, hmany=None):
    assert("idx3" in file_name)
    result = []
    if ".com" in file_name:
        file = gzip.open(requests.get(file_name, stream=True).raw, 'rb')
    else:
        file = open(file_name, "rb")
    byte = int.from_bytes(file.read(4), byteorder='big')    # magic number
    no = int.from_bytes(file.read(4), byteorder='big')      # number of images
    rows = int.from_bytes(file.read(4), byteorder='big')    # number of rows
    cols = int.from_bytes(file.read(4), byteorder='big')    # number of columns
    for img in range(no):
        temp_img = [[256 for r in range(rows)] for c in range(cols)]
        for c in range(cols):
            for r in range(rows):
                temp_img[c][r] = int.from_bytes(file.read(1), byteorder='big') / 255
        result.append(temp_img)
        if img % 100 == 0:
            print(img)
        if img == hmany-1:
            break
    file.close()
    return np.array(result)


def make_pair(img, label):
    return img_to_array(img), label


def create_data(imgarr, labelarr):
    data = []
    if len(imgarr) != len(labelarr):
        raise ValueError("Two datasets doesn't have same length!")
    for i in range(len(imgarr)):
        data.append(make_pair(imgarr[i], labelarr[i]))
    return data

def divide_train_test(data, index):
    return data[:index],data[index:]

def img_to_array(img_result):
    one_big_chungus = []
    for line in img_result:
        one_big_chungus += list(line)
    return one_big_chungus


def array_to_img(ar, size=28):
    matrix = []
    for i in range(size):
        matrix.append(ar[size*i:size*(i+1)])
    return matrix


def show_img_arr(arr):
    plt.matshow(array_to_img(arr))
    plt.colorbar()
    plt.show()
    plt.clf()
