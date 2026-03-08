from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np

def probabilitificator(labels):
    new_labels = []
    for label in labels:
        new_label = np.zeros((10,1),dtype=np.float32)
        new_label[label] = 1.0
        new_labels.append(new_label)
    return new_labels

def normalize(images):
    # make the data in the range [0,1]
    # hard coded 0-255 because i can
    images = np.array(images,dtype=np.float32) #ensure its a np array to use np vector operations
    images = images/255.0
    return images



train_dataset = MNIST(root = './data',train=True,download=True)
test_dataset = MNIST(root = './data',train=False,download=True)

train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()

train_images_flattened = np.array(train_images,dtype=np.float32).reshape(len(train_images),-1)

train_images_flattened_normalized = normalize(train_images_flattened)

test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

test_images_flattened_normalized = normalize(np.array(test_images.reshape(len(test_images),-1),dtype=np.float32))

#for convertint it to .npz
train_images, train_labels = train_images_flattened_normalized, probabilitificator(train_labels)

test_images,test_labels = test_images_flattened_normalized, probabilitificator(test_labels)

def make_Mnist_data_file():
    print(f'writing...')
    np.savez_compressed('Mnist_data.npz',
                        train_images=train_images,
                        train_labels=train_labels,
                        
                        test_images=test_images,
                        test_labels=test_labels)

    print(f'wrote data successfully to .npz')

def Load_data():
    from os import path

    if path.exists("Mnist_data.npz"):
        print(f'Mnist_data file already exists => not doing anything')
    else:
        try:
            make_Mnist_data_file()
        except Exception as e:
            print(f'Failed to make file because: {e}')
