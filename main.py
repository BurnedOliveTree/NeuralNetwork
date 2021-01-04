import numpy as np
import neonetwork as nn

def load_labels(file_name):
    assert("idx1" in file_name)
    result = []
    with open(file_name, "rb") as file:
        byte = int.from_bytes(file.read(4), byteorder='big')    # magic number
        no = int.from_bytes(file.read(4), byteorder='big')      # number of images
        for img in range(no):
            result.append(int.from_bytes(file.read(1), byteorder='big'))
    return np.array(result)


def load_images(file_name):
    assert("idx3" in file_name)
    result = []
    with open(file_name, "rb") as file:
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
            if img % 1000 == 0:
                print(img)
    return np.array(result)

def img_to_array(img_result):
    one_big_chungus = []
    for line in img_result:
        one_big_chungus+=list(line)
    return one_big_chungus

if __name__ == "__main__":
    labels = load_labels("train-labels.idx1-ubyte")
    images = load_images("train-images.idx3-ubyte")
    net = nn.Network(img_to_array(images[0]))
    net.go_through()
    print(net.get_out())

