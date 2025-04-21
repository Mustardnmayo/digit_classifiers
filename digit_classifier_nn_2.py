import numpy as np
import time 



#in this project we are using np.float32 because my computer is not a 'powerhouse', and I want to attempt to be effecient
#chat gpt wrote this function, but I do atleast understand it
def enforce_float32(func):
    def wrapper(*args,**kwargs):
        result = func(*args,**kwargs)
        if isinstance(result,np.ndarray) and result.dtype != np.float32:
            #print(f'function {func.__name__} return {result.dtype}, auto casting to np.float32')
            return result.astype(np.float32)
        elif isinstance(result,float) and type(result) != np.float32:
            #print(f'function {func.__name__} return a python float(64), auto casting to np.float32')
            return result.astype(np.float32)
        return result
    return wrapper

'''
how to use
@enforce_float32
def safe_mean(arr):
    return np.mean(arr)
'''
#defining all the safe functions
@enforce_float32
def safe_nan_to_num(num):
    return np.nan_to_num(num)
@enforce_float32
def safe_dot(*args):
    return np.dot(*args)
@enforce_float32
def safe_randn(*args):
    return np.random.randn(*args)
#-----------------------------------------------------|getting the data|---------------------------------------------------------------
from Mnist_data_loader import Load_data

Load_data()

data = np.load('Mnist_data.npz',allow_pickle=True)
train_data = list(zip(data['train_images'],data['train_labels']))
test_data = list(zip(data['test_images'],data['test_labels']))
print(f'{len(train_data) = } | image, label lengths {str([len(item) for item in train_data[0]])}')
print(f'{len(test_data) = }  | image, label lengths {str([len(item) for item in test_data[0]])}')
#testing
#print(f'{train_data[0] = }\n{test_data[0] = }')

class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        '''
        returns cost of output a for desiered output y
        '''
        return np.sum(safe_nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)),dtype=np.float32)
    
    def delta(z,a,y):
        '''
        returns the error delta from the output layer
        z is here for consistency w/ other similar calls
        '''
        return(a-y)
#-----------------------------------------------------^the cost class^-----------------------------------------------------
def bias_generator(sizes):
    biases = []
    for index,layer_len in enumerate(sizes):
        if index == 0:
            continue
        biases.append(safe_randn(layer_len).reshape(-1,1))
    print(f'shapes of bias vectors: \n\t{str([b.shape for b in biases ])}\n')
    return biases

def weight_generator(sizes):
    weights = []
    for x,y in zip(sizes[:-1], sizes[1:]):
        weights.append(safe_randn(y,x))
    print(f'shapes of weight matrices:\n\t{str([w.shape for w in weights ])}\n')
    return weights

class neural_network(object):
    @staticmethod
    def sigmoid(z,derivative=False):
        # this will work w/ np arrays
        if derivative:
            z = neural_network.sigmoid(z) 
            return (z*(1-z))
        #safe_exp has default base of e
        return 1/(1 + np.exp(-z,dtype=np.float32))
    

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
        nabla_b = [np.zeros(b.shape,dtype=np.float32) for b in self.biases]
        nabla_w = [np.zeros(w.shape,dtype=np.float32) for w in self.weights]
        #first we need to make lists of:
        #the error of each layer
        #the activations of each layer
        weighted_sums = []
        current_activation = image.reshape(-1,1)
        #print(f'{current_activation.shape = } (should be (784,1))')
        
        activations = [current_activation]

        for biases,weights in zip(self.biases,self.weights):
            #calculate z
            z = safe_dot(weights,current_activation) + biases
            current_activation = neural_network.sigmoid(z) #forward data

            weighted_sums.append(z)
            activations.append(current_activation)

        #backwards pass
        #first layer
        #will be calling error delta, to avoid naming issues
        delta = self.cost.delta(weighted_sums[-1],activations[-1],label)
        nabla_b[-1] = delta
        nabla_w[-1] = safe_dot(delta,activations[-2].T)

        for l in range(2,len(self.sizes)):
            z = weighted_sums[-l]
            sp = neural_network.sigmoid(z,derivative=True)
            #print(f'{sp.shape = }')
            #print(f'{self.weights[-l+1].T.shape = }\n{delta.shape = }')
            delta = safe_dot(self.weights[-l+1].T,delta) * sp

            #print(f'{delta.shape = } \n{activations[-l-1].T.shape = }')
            nabla_b[-l] = delta
            nabla_w[-l] = safe_dot(delta,activations[-l-1].T)
        return (nabla_b,nabla_w)

    '''def back_prop(self,image,label):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]

        activation = image
        activations = [activation]

        zs = [] #weighted inputs
        for b,w in zip(self.biases,self.weights):
            z = safe_dot(activation,w)+b
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
        nabla_w[-1] = safe_dot(error,np.transpose(activations[-1])) # -1 in activations should be -2

        for layer in range(2,len(self.sizes)):
            z = zs[-layer]
            sp = neural_network.sigmoid(z,derivative=True)
            #error = safe_dot(np.transpose(self.weights[-layer+1]),error) * sp
            error = safe_dot(error,np.transpose(self.weights[-layer+1])) *sp

            nabla_b[-layer] = error 
            nabla_w[-layer] = safe_dot(error,np.transpose(-layer-1))

        return (nabla_b,nabla_w)'''

    def SDG(self,training_dataset,epochs,batch_size,learning_rate):
        for i in range(epochs-1): #an epoch is the entire dataset
            print(f'starting epoch {i+1}')
            np.random.shuffle(training_dataset)
            ra = range(0,len(training_dataset),batch_size)
            #print(list(ra))
            batches = [training_dataset[k:k+batch_size] for k in ra]
            for batch in batches:
                self.update_batch(batch,learning_rate)
            print(f'completed epoch {i+1} out of {epochs}')
            print(f'accuracy is {self.testing(test_data)}%')

    def forward(self,input): #need this funciton for later(and testing), not for backprop
        current_activation = input.reshape(-1,1)
        #print(f'{current_activation.shape = }')
        for biases,weights in zip(self.biases,self.weights):
            current_activation = neural_network.sigmoid(safe_dot(weights,current_activation) + biases)
        return current_activation

    def testing(self,test_data):
        test_results = [
            (np.argmax(self.forward(image)),np.argmax(label))
              for image,label in test_data]
        return sum(int(x==y) for x,y in test_results)*100 / len(test_data)
#--------------------------------------------------------------------------------------
start = time.time()

network = neural_network([28*28,256,64,32,10],CrossEntropyCost)
#training
network.SDG(train_data,10,64,0.01)
accuracy = network.testing(test_data)
print(f'accuracy = {accuracy:.3f}%')

print(f'total time taken: {(time.time()- start):.3f}s')
