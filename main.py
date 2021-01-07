import neonetwork as nn
import dataloader as dl


def foo(net, images, labels, which):
    net.set_input(dl.img_to_array(images[which]))
    net.forward_prop()
    # dl.show_img_arr(dl.img_to_array(images[which]))
    print(net.get_out())
    print(labels[which])


if __name__ == "__main__":
    labels = dl.load_labels("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", 60000)
    images = dl.load_images("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", 60000)
    d = dl.create_data(images,labels)
    train, test = dl.divide_train_test(d, 48000)

    net = nn.Network([0 for i in range(784)])
    net.train(train, 10)
    net.test(test)


