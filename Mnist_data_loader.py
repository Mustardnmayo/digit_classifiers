from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np

def probabilitificator(labels):
    new_labels = []
    for label in labels:
        new_label = np.zeros((10,1),dtype=np.float64)
        new_label[label] = 1.0
        new_labels.append(new_label)
    return new_labels

def normalize(images):
    # make the data in the range [0,1]
    # hard coded 0-255 because i can
    images = np.array(images) #ensure its a np array to use np vector operations
    images = images/255.0
    #images = 2*images -1
    return images



train_dataset = MNIST(root = './data',train=True,download=True)
test_dataset = MNIST(root = './data',train=False,download=True)

train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()

train_images_flattened = np.array(train_images).reshape(len(train_images),-1)
#print(f'{train_images_flattened[0].shape = }\n{train_images_flattened[0] = }')

train_images_flattened_normalized = normalize(train_images_flattened)

#train_data = list(zip(train_images_flattened_normalized,probabilitificator(train_labels)))
#list of tuples, (image,labels_probability_vector)

test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

test_images_flattened_normalized = normalize(np.array(test_images.reshape(len(test_images),-1)))
#test_data = list(zip(test_images_flattened_normalized,probabilitificator(test_labels)))

#for convertint it to .npz
train_images, train_labels = train_images_flattened_normalized, probabilitificator(train_labels)

test_images,test_labels = test_images_flattened_normalized, probabilitificator(test_labels)

raise ZeroDivisionError
np.savez_compressed('Mnist_data.npz',
                    train_images=train_images,
                    train_labels=train_labels,
                    
                    test_images=test_images,
                    test_labels=test_labels)

print(f'wrote data successfully to .npz')

'''
#convert the data to json seralizable data
try:
    raise ZeroDivisionError
    file = open(r'Mnist_data.json','w')
    json.dump({
            'train':train_data_seralizable,
            'test':test_data_seralizable
    },file,indent=2)
    print(f'dumped data successfuly')
except Exception as e:  
    print(f'failed:\n{e}')
'''