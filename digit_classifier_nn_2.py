from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import time 

from Mnist_data_loader import Load_data

Load_data()


#-----------------------------------------------------|getting the data|---------------------------------------------------------------
data = np.load('Mnist_data.npz',allow_pickle=True)
train_data = list(zip(data['train_images'],data['train_labels']))
test_data = list(zip(data['test_images'],data['test_labels']))
#testing
#print(f'{train_data[0] = }\n{test_data[0] = }')

class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        '''
        returns cost of output a for desiered output y
        '''
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    def delta(a,y):
        '''
        returns the error delta from the output layer
        '''
        return(a-y)

#-----------------------------------------------------^the cost class^-----------------------------------------------------
def bias_generator(sizes):
    biases = []
    for index,layer_len in enumerate(sizes):
        if index == 0:
            continue
        biases.append(np.random.randn(layer_len))
    print(f'shapes of bias vectors: \n\t{str([b.shape for b in biases ])}\n')
    return biases

def weight_generator(sizes):
    weights = []
    for x,y in zip(sizes[:-1], sizes[1:]):
        weights.append(np.random.randn(x,y))
    print(f'shapes of weight matrices:\n\t{str([w.shape for w in weights ])}\n')
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
    

    def __init__(self,sizes,cost_function):
        assert(cost_function.delta)
        self.cost = cost_function

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
        #first we need to make lists of:
        #the error of each layer
        #the activations of each layer
        weighted_sums = np.array([])
        
        current_activation = image
        activations = np.array([current_activation])

        for biases,weights in zip(self.biases,self.weights):
            #calculate z
            weighted_sum = np.dot(current_activation,weights)+biases
            current_activation = neural_network.sigmoid(weighted_sum)
            weighted_sums = np.append(weighted_sums,weighted_sum)
            activations = np.append(activations,current_activation)

        print(f'{weighted_sums.ndim = } {weighted_sums[0] = }    \n  {activations.ndim = } {activations[0] = }')
        # 


        return None


    '''def back_prop(self,image,label):
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
        print(f'error shape:{error.shape} | transposed activations[-1] {np.transpose(activations[-1]).shape}')
        nabla_w[-1] = np.dot(error,np.transpose(activations[-1])) # -1 in activations should be -2

        for layer in range(2,len(self.sizes)):
            z = zs[-layer]
            sp = neural_network.sigmoid(z,derivative=True)
            #error = np.dot(np.transpose(self.weights[-layer+1]),error) * sp
            error = np.dot(error,np.transpose(self.weights[-layer+1])) *sp

            nabla_b[-layer] = error 
            nabla_w[-layer] = np.dot(error,np.transpose(-layer-1))

        return (nabla_b,nabla_w)'''

    def SDG(self,training_dataset,epochs,batch_size,learning_rate):
        for i in range(epochs-1): #an epoch is the entire dataset
            print(f'starting epoch {i}')
            np.random.shuffle(training_dataset)
            ra = range(0,len(training_dataset),batch_size)
            #print(list(ra))
            batches = [training_dataset[k:k+batch_size] for k in ra]
            for batch in batches:
                self.update_batch(batch,learning_rate)
            print(f'completed epoch {i+1} out of {epochs}')


    def testing(self,test_data):
        test_results = [(np.argmax(self.forward(image)),label) for image,label in test_data]
        return sum(int(x==y) for x,y in test_results)
#--------------------------------------------------------------------------------------
start = time.time()

network = neural_network([28*28,128,32,10],CrossEntropyCost)
#training
network.SDG(train_data,5,64,0.01)
network.testing(test_data)


print(f'total time taken: {time.time()- start}s')
