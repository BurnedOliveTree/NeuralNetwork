import neonetwork as nn
import dataloader as dl

def foo(images, labels, which):
    neti.set_input(dl.img_to_array(images[which]))
    neti.forward_prop()
    #dl.show_img_arr(dl.img_to_array(images[which]))
    print(neti.get_out())
    print(labels[which])

if __name__ == "__main__":
    labels = dl.load_labels("train-labels.idx1-ubyte", 2500)
    images = dl.load_images("train-images.idx3-ubyte", 2500)
    d = dl.create_data(images,labels)
    train, test = dl.divide_train_test(d, 2000)

    neti = nn.Network([0 for i in range(784)])
    neti.train(train, 15)
    neti.test(test)

