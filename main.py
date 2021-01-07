import neonetwork as nn
import dataloader as dl


def foo(images, labels, which):
    net.set_input(dl.img_to_array(images[which]))
    net.forward_prop()
    # dl.show_img_arr(dl.img_to_array(images[which]))
    print(net.get_out())
    print(labels[which])


if __name__ == "__main__":
    # labels = dl.load_labels("train-labels.idx1-ubyte", 5000)
    # images = dl.load_images("train-images.idx3-ubyte", 5000)
    labels = dl.load_labels("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", 2000)
    images = dl.load_images("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", 2000)

    net = nn.Network([0 for i in range(784)])

    data = dl.create_data(images, labels)
    net.train(data, 5)
    net.test(data)

    foo(images, labels, 0)
    foo(images, labels, 1)
    foo(images, labels, 2)
    foo(images, labels, 3)
    foo(images, labels, 4)
    foo(images, labels, 5)
    foo(images, labels, 6)
    foo(images, labels, 7)

