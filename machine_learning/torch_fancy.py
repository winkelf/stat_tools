# example making new class predictions for a classification problem
import torch
import torch.nn as torch_nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from functools import partial

############################## Parameters ###############################################################################################
nFeatures         = 2
nSamples          = 100
nTestingSamples   = 100
plotInputs        = False

Layers       = [15,10,1]  # Last layer must have 1 node for binary classification
batchSize    = 10 
nEpochs      = 100
learningRate = 0.001

############################## Functions ################################################################################################
def plotter(feat, sample):
    l = sample[:, feat]
    bins = np.linspace(-10,10, 100)
    plt.hist(l, bins)
    return plt.show()

def toTensor(z):
    """Convert arrays into torch tensors"""
    z = torch.tensor(z)
    z = z.to(torch.float32)
    return z

def batch(inputSample, start, stop):
    """"Explicitely build batch"""
    arrayList=[]
    for i in range(start, stop):
        arrayList.append( inputSample[i] )

    # build a compact 2d array by stacking individual features:
    inputs = torch.stack(arrayList)
    return inputs

def lossFunction():
    """ Binary cross entropy loss function """
    dE = torch_nn.BCELoss()
    return dE

############################## Classes ##################################################################################################
" We write a Module to represent mlp using several nn.Linear and a nn.Sequential, following something like :"   
"                      mlp = Sequential( Linear(n0,n1), activation(),                                       "
"                                        Linear(n1,n2), activation(),                                       "
"                                        Linear(n2,n3), activation(),                                       "
"                                        ...)                                                               "

class MLP(torch.nn.Module):
    def __init__(self, numInput, layers=[], activation=torch.nn.Mish(), last_activation=None):
        """ Mish(x) = x*tanh(softplus(x)), just an activation function. It's like a smooth relu """ 
        super().__init__()
        nin = numInput
        seq = []
        for nout in layers:
            seq += [ torch.nn.Linear(nin, nout), activation ]
            nin = nout
        if last_activation is None: last_activation=activation
        seq[-1] = last_activation

        self.model = torch.nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)

class lastActivation(torch.nn.Module):
    """Define an activation giving values between [0,1]"""
    def forward(self,x):
        return torch.sigmoid(x)

class NN(torch.nn.Module):
    """The NN itself"""
    def __init__(self, ):
        super().__init__()
        #self.norm     = Normalization(*buildSCalesOffsets())
        self.preMLP   = MLP(nFeatures, Layers, last_activation=lastActivation() )

    def forward(self, features):
        y = self.preMLP(features)
        return y

############################## Body #####################################################################################################

# quickly generate 2d classification dataset
X, y = make_blobs(n_samples    = nSamples,  # Esta funcion genera array de arrays (X e y) de tamanho n_samples de v.a gaussianas. 
                  centers      = 2,         # centers me cambia "el numero de gaussianas", si centers=2 tengo 2 gaussianas de distintas esperanzas 
                  n_features   = nFeatures, # Si n_features=1, los arrays X e y son simplemente listas de n_samples elementos
                  random_state = 1)         # Si n_features=2, los arrays X e y son listas de vectores de 2 componentes 

# Scale data to [0,1]
scalar = MinMaxScaler()   
scalar.fit(X)             
X = scalar.transform(X)   

# Plot inputs
if (plotInputs == True): plotter(0, X)

# Convert inputs to tensors
X = toTensor(X)
y = toTensor(y)
y = y.view(nSamples, 1)

# Initialize instance of NN class
nn = NN()

# Function for training
def trainingLoop(nn, batchSize, nepochs, opti, lossFunc):
    Ntot = nSamples

    # opti is a function returning a pytorch optimizer
    optimizer = opti(nn.parameters())

    for epoch in range(nepochs):
        for i,b in enumerate(range(0, Ntot, batchSize)):
            stop = min(Ntot, b+batchSize)
            #print("epoch",epoch, "step", i)

            # create a batch 
            feat_b = batch(X, b, stop)

            # model prediction
            modelPrediction = nn(feat_b)

            # retrieve the labels for this batch
            labels = y[b:stop]

            # calculate the loss
            loss = lossFunc(modelPrediction, labels)
            loss = loss.mean()

            # perform an optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i%100==0):
                print('loss ',loss)

    return nn


# Run the training (`partial` is a trick to predefine which parameters are going to be passed to the optimizer)
trainingLoop(nn, batchSize, nEpochs, partial(torch.optim.NAdam, lr=learningRate), lossFunction())

# Compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = nn(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# Plot scoring
s = []
b = []
for i in range(nSamples):
    if (y[i,0].numpy() == 0): b.append( nn(X)[i,0].detach().numpy() )
    if (y[i,0].numpy() == 1): s.append( nn(X)[i,0].detach().numpy() )

bins = np.linspace(0, 1, 81)

plt.hist([s,b], bins, label=["S", "B"])
plt.legend(loc='upper right')
plt.show()

## new instances where we do not know the answer
#Xnew, ynew = make_blobs(n_samples=nTestingSamples, centers=2, n_features=2, random_state=1)
##Xnew = scalar.transform(Xnew)
#
#Xnew = toTensor(Xnew)
#ynew = toTensor(ynew)
#ynew = ynew.view(nTestingSamples,1)

