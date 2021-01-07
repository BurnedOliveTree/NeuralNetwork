import neonetwork as nn
import dataloader as dl

def foo(images, labels, which):
    neti.set_input(dl.img_to_array(images[which]))
    neti.forward_prop()
    #dl.show_img_arr(dl.img_to_array(images[which]))
    print(neti.get_out())
    print(labels[which])

if __name__ == "__main__":
    labels = dl.load_labels("train-labels.idx1-ubyte", 2000)
    images = dl.load_images("train-images.idx3-ubyte", 2000)

    neti = nn.Network([0 for i in range(784)])
    neti.train(dl.create_data(images,labels), 5) # Dziwne u mnie działa
    neti.test(dl.create_data(images,labels))

    foo(images, labels, 0)
    foo(images, labels, 1)
    foo(images, labels, 2)
    foo(images, labels, 3)
    foo(images, labels, 4)
    foo(images, labels, 5)
    foo(images, labels, 6)
    foo(images, labels, 7)

