from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np


def normalize(images):
    # make the data in the range [0,1]
    # hard coded 0-255 because i can
    images = np.array(images) #ensure its a np array to use np vector operations
    images = images/255.0
    #images = 2*images -1
    return images

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

def vector_average(list_of_vectors):
    list_of_vectors = np.array(list_of_vectors) #to be safe
    for vector in list_of_vectors:
        assert((len(vector) != len(list_of_vectors[0])) and f"Not all vectors are the same length \n vector_average {len(list_of_vectors[0])} != {len(vector) = } at index {list_of_vectors.index(vector)}")
    return np.mean(list_of_vectors,0)

#-----------------------------------------------------THE FUNCTION LINE---------------------------------------------------------------

train_dataset = MNIST(root = './data',train=True,download=True)
test_dataset = MNIST(root = './data',train=False,download=True)

train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()

train_images_flattened = np.array(train_images).reshape(len(train_images),-1)
print(f'{train_images_flattened[0].shape = }\n{train_images_flattened[0] = }')


train_images_flattened_normalized = normalize(train_images_flattened)

train_data = list(zip(train_images_flattened_normalized,train_labels))

#data_iter = iter(train_data) # a list of tuples w/ image first, then lable value
#image,label = next(data_iter)
#print(f'{image = }\n {int(label) = }')


'''
class neural_network(object): #make this work (duh)
    def __init__(self,input_layer,output,hidden_layers=np.array([])): #hidden_layers is a list of lengths

        self.input_layer = input_layer

        self.hidden_layers = hidden_layers
        
        self.output = output

        self.layers = np.array([]) #(will be) an array of lengths

        np.append(self.layers,input_layer)

        for L in self.hidden_layers:
            np.append(self.layers,L)

        np.append(self.layers, output)

        print(f'{self.layers = }')

        self.layers_linalg = np.array([]) # this is the list that actually contains all the layers

        for layer in self.layers:
            np.append(self.layers_linalg,neural_network.layer.__init__(self,len(layer),layer))


        def forward(self):
            for i in range(len(self.layers_linalg)-1):
                neural_network.layer.forward(self.layers_linalg[i],self.layers_linalg[i+1])


        def __str__(self):
            ret = f'a neural network with an input layer of {self.layers[0]}\n'
            ret += f'and hidden layers of {str([l for l in self.layers[1:-2]])}\n'
            ret += f'and an output layer of {self.layers[-1]}\n'
            ret += f'layers :\n'
            ret += f'{str([str(l) + '\n' for l in self.layers])}\n'
            return ret
#-------------------------------------------------THE NETWORK LINE-------------------------------------------------------
    class layer(object):
        def __init__(self,network,layer_id,neurons):
            self.network = network
            self.layer_id = layer_id

            self.neurons = np.array([])
            for index in range(neurons):
                np.append(self.neurons,neural_network.neuron.__init__(self.network,self.layer_id))

        @staticmethod
        def forward(first_layer, second_layer):
            print(f'forwarding from layer {first_layer.layer_id} to layer {second_layer.layer_id}')
            print(first_layer)
            print(second_layer)
            

        def __str__(self):
            ret = f'layer: {self.layer_id = } \n in network: {self.network = } \n with {len(self.neurons)} neurons'
#----------------------------------------------THE LAYER LINE------------------------------------------------------------
    class neuron(object):
        def __init__(self,network,layer,weight=None,bias=None):
            self.network = network
            self.layer = layer   
            #random initalization
            if not weight:
                pass
            else:
                self.weight = weight
            if not bias:
                pass
            else:
                self.bias = bias

        def __str__(self):
            ret = f'neuron in {self.network = } \n {self.layer = } \n {self.weight = } \n {self.bias = }'
            return ret
#----------------------------------------------THE NEURON LINE----------------------------------------------



net = neural_network((28*28),10,np.array([128,32]))
print(f'{net = }')
'''


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
        assert(isinstance(sizes, np.array) and 'sizes needs to be a numpy array')
        self.sizes = sizes
        self.biases = [(np.random.randn(y,1) for y in sizes[1:])] # no biases for input layer
        self.weights = [(np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:]))] #weights matrix for layers
        #the above line indexs over lists [all sizes except last] and [all sizes except first], to create 
        # matrices of layer X nextlayer
        print(f'{self.biases[0] = }')
        print(f'{self.weights[0] = }')

    def feed_forward(self,input):
        for b,w in zip(self.biases,self.weights):
            input = neural_network.sigmoid(np.dot(w,input)+b) 
            #dot product is [1,2]dot[3,4] = sum[a_i * b_i] = 3+8 = 11
        return input
    
    def cost_derivative(self,output_activations,y):
        return output_activations-y #I THINK THIS IS WRONG

    def update_mini_batch(mini_batch,learning_rate):
        '''
        mini batch is list of tuples (image,label)
        '''
        for image,label in mini_batch:
            pass


    def SDG(self,training_data,minibatch_size,epochs,leanring_rate):
        n = len(training_data)
        for i in range(epochs):
            training_data = np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+minibatch_size] for k in range(0,n,minibatch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,leanring_rate)
                print(f'epoch {i} complete')


    
    def testing(self,testing_data,batch_size):
        pass



