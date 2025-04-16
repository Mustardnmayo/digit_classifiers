from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import time 


'''def normalize(images):
    # make the data in the range [0,1]
    # hard coded 0-255 because i can
    images = np.array(images) #ensure its a np array to use np vector operations
    images = images/255.0
    #images = 2*images -1
    return images
'''
'''
def vector_average(list_of_vectors):
    #averager function takes in list returns float average
    def average(list):
        a = 0
        for element in list:
            a += element
        return a/len(list)

    #check that all vectors are the same length:
    length = len(list_of_vectors[0])
    for v in range(0,len(list_of_vectors)):
        assert((len(list_of_vectors[v]) != length) and f"Not all vectors are the same length vector_average {length} != {len(list_of_vectors[v]) = } at index {v}")
    
    sum_vector = np.empty(length)
    assert(len(sum_vector) == len(list_of_vectors[0]))
    
    storage = np.empty(len(list_of_vectors))
    for index in len(0,list_of_vectors):
        for vector in list_of_vectors:
            np.append(storage, vector[index])
        sum_vector[index] =  average(storage))
        storage = np.empty(len(list_of_vectors))

    return sum_vector
'''
'''
def probabilitificator(labels):
    new_labels = []
    for label in labels:
        new_label = np.zeros((10,1),dtype=np.float64)
        new_label[label] = 1.0
        new_labels.append(new_label)
    return new_labels
'''
#-----------------------------------------------------|getting the data|---------------------------------------------------------------
data = np.load('Mnist_data.npz',allow_pickle=True)
train_data = list(zip(data['train_images'],data['train_labels']))
test_data = list(zip(data['test_images'],data['test_labels']))
#testing
#print(f'{train_data[0] = }\n{test_data[0] = }')


#-----------------------------------------------------^the shitty class^-----------------------------------------------------
def bias_generator(sizes):
    biases = []
    for index,layer_len in enumerate(sizes):
        if index == 0:
            continue
        biases.append(np.random.randn(layer_len))
    #print(f'shapes of bias vectors \n\t{str([b.shape for b in biases ])}\n')
    return biases

def weight_generator(sizes):
    weights = []
    for x,y in zip(sizes[:-1], sizes[1:]):
        weights.append(np.random.randn(x,y))
    #print(f'shapes of weight matrices:\n\t{str([w.shape for w in weights ])}\n')
    return weights

class neural_network(object):

    @staticmethod
    def sigmoid(z,derivative=False):
        # this will work w/ np arrays
        if derivative:
            z = neural_network.sigmoid(z) 
            return (z*(1-z))
        #np.exp has default base of e
        return 1/(1 + np.exp(-z))
    

    def __init__(self,sizes):
        assert((isinstance(sizes, list) or isinstance(sizes,np.array)) and 'sizes needs to be a numpy array')
        self.sizes = sizes
        '''
        #self.biases = [(np.random.randn(y,1) for y in sizes[1:])] # no biases for input layer
        #self.weights = [(np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:]))] #weights matrix for layers
        #the above line indexs over lists [all sizes except last] and [all sizes except first], to create 
        # matrices of layer by nextlayer
        '''
        self.biases = bias_generator(sizes)
        self.weights = weight_generator(sizes) #REMEMBER FOR LATER: mn * np = mp
        
    def forward(self,input): #need this funciton for later(and testing), not for backprop
        assert(input.shape == (1,784))    
        for layer in range(len(self.sizes)-1):
            b = self.biases[layer]
            #print(f'{b.shape = }')
            w = self.weights[layer]
            #print(f'{w.shape = }')
            #print(f'operation: ({input.shape} dot {w.shape}) + {b.shape}')
            input = neural_network.sigmoid(np.dot(input,w)+b)
            #print(f'{input.shape = }')
        
        return input
    
    def update_batch(self,batch,learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]

        for image,label in batch:
            delta_nabla_b,delta_nabla_w = self.back_prop(image,label)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]

        eta_bar = learning_rate/len(batch)
        self.weights = [w-eta_bar*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-eta_bar*nb for b,nb in zip(self.biases,nabla_b)]

    def back_prop(self,image,label):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]

        activation = image
        activations = [activation]

        zs = [] #weighted inputs
        for b,w in zip(self.biases,self.weights):
            z = np.dot(activation,w)+b
            zs.append(z)
            activation = neural_network.sigmoid(z)
            activations.append(activation)
        print(f'activation shapes:')
        for a in activations:
            print(f'\t{a.shape}')
        #backwards pass
        #first layer
        error = neural_network.cost_derivative(activations[-1],label) * neural_network.sigmoid(zs[-1],derivative=True)
        
        #i get why you wouldnt typically call this error, but you cant stop me
        nabla_b[-1] = error
        print(f'error shape:{error.shape} | transposed activations[-2] {np.transpose(activations[-2]).shape}')
        nabla_w[-1] = np.dot(error,np.transpose(activations[-2]))

        for layer in range(2,len(self.sizes)):
            z = zs[-layer]
            sp = neural_network.sigmoid(z,derivative=True)
            error = np.dot(np.transpose(self.weights[-layer+1]),error) * sp

            nabla_b[-layer] = error 
            nabla_w[-layer] = np.dot(error,np.transpose(-layer-1))

        return (nabla_b,nabla_w)

    def SDG(self,training_dataset,epochs,batch_size,learning_rate):
        for i in range(epochs-1): #an epoch is the entire dataset
            np.random.shuffle(training_dataset)
            ra = range(0,len(training_dataset),batch_size)
            print(list(ra))
            batches = [training_dataset[k:k+batch_size] for k in ra]
            for batch in batches:
                self.update_batch(batch,learning_rate)
            print(f'completed epoch {i+1} out of {epochs}')

    @staticmethod
    def cost_derivative(output_activations,expected_activations):
        return (output_activations-expected_activations)

    def testing(self,test_data):
        test_results = [(np.argmax(self.forward(image)),label) for image,label in test_data]
        return sum(int(x==y) for x,y in test_results)
#--------------------------------------------------------------------------------------
start = time.time()

network = neural_network([28*28,128,32,10])
#training
network.SDG(train_data,5,64,0.01)
network.testing(test_data)


print(f'total time taken: {time.time()- start}s')
