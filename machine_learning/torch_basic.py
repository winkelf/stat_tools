import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs

############################## Parameters ###############################################################################################
nFeatures         = 2
nSamples          = 100

batchSize    = 10
nEpochs      = 40
learningRate = 0.001
########################################################################################################################################

X, y = make_blobs(n_samples    = nSamples,  # Esta funcion genera array de arrays (X e y) de tamanho n_samples de v.a gaussianas. 
                  centers      = 2,         # centers me cambia "el numero de gaussianas", si centers=2 tengo 2 gaussianas de distintas esperanzas 
                  n_features   = nFeatures, # Si n_features=1, los arrays X e y son simplemente listas de n_samples elementos
                  random_state = 1)         # Si n_features=2, los arrays X e y son listas de vectores de 2 componentes 

# Scale data to [0,1]
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
model = nn.Sequential(
    nn.Linear(nFeatures, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(nEpochs):
    for i in range(0, len(X), batchSize):
        Xbatch = X[i:i+batchSize]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batchSize]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy
y_pred   = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
