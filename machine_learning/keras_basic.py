# example making new class predictions for a classification problem
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# quickly generate 2d classification dataset
X, y = make_blobs(n_samples=100,   # Esta funcion genera array de arrays (X e y) de tamanho n_samples de v.a gaussianas. 
                  centers=2,       # centers me cambia "el numero de gaussianas", si centers=2 tengo 2 gaussianas de distintas esperanzas 
                  n_features=2,    # Si n_features=1, los arrays X e y son simplemente listas de n_samples elementos
                  random_state=1)  # Si n_features=2, los arrays X e y son listas de vectores de 2 componentes 


scalar = MinMaxScaler()   #  Estas lineas son para llevar el eje X de mi histograma al rango [0,1]
scalar.fit(X)             #
X = scalar.transform(X)   #

# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=600) #, verbose=0)

# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)

# make a prediction
ynew = (model.predict(Xnew) > 0.5).astype("int32")

# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
    if i>10: break

# new instances where we do not know the answer
Xnew, ynew = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)

# make a prediction
ynew_scores = model.predict(Xnew)  # Raw scores (probabilities)
ynew_pred = (ynew_scores > 0.5).astype("int32")

# plot scores for truth signal and background (training)
y_train_scores = model.predict(X)
y_train_scores = (y_train_scores > 0.5).astype("int32")

plt.hist(y_train_scores[y == 0], bins=np.linspace(0,1,100), alpha=1, label='Background (train)', color='blue')
plt.hist(y_train_scores[y == 1], bins=np.linspace(0,1,100), alpha=0.2, label='Signal (train)', color='orange')

plt.hist(ynew_scores[ynew == 0], bins=np.linspace(0,1,100), alpha=1, label='Background (test)', color='green')
plt.hist(ynew_scores[ynew == 1], bins=np.linspace(0,1,100), alpha=0.2, label='Signal (test)', color='red')

# Add labels and legend
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Model Score Distribution')
plt.legend()
#plt.yscale('log')
plt.show()    
