#####################################################################################################
# Error propagation script, taken from statistics course. For now, only in Argentinean Spanish ***  #
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats as stats
from scipy.optimize import curve_fit

# Los datos ----------------------------------------------------------
x = [2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00]
y = [2.78, 3.29, 3.29, 3.33, 3.23, 3.69, 3.46, 3.87, 3.62, 3.40, 3.99]

sigma = 0.3
N     = len(x)
# --------------------------------------------------------------------

delta = N*sum( (np.array(x))**2 ) - sum(np.array(x))**2
a1    = ( sum( (np.array(x))**2 )*sum(y) - sum(x)*(sum(np.array(x)*np.array(y))) )/delta
a2    = ( N*sum(np.array(x)*np.array(y)) - sum(x)*sum(y) )/delta

def recta(x):
    y = a1 + a2*x
    return y

cov = [[ [], [] ], [ [], [] ]]
cov[0][0] = ((sigma**2)/delta)*sum( (np.array(x))**2 )  
cov[0][1] = ((sigma**2)/delta)*(-sum(x))
cov[1][0] = ((sigma**2)/delta)*(-sum(x))
cov[1][1] = ((sigma**2)/delta)*N

err_a1 = np.sqrt(cov[0][0])
err_a2 = np.sqrt(cov[1][1])

print(f"a1 = {round(a1,3)}+-{round(err_a1,3)}")
print(f"a2 = {round(a2,3)}+-{round(err_a2,3)}")

def bandaError(X):
    err = cov[0][0] + cov[1][1]*X*X + 2*X*cov[0][1]
    err = np.sqrt(err)
    return err

def bandaErrorMala(X):
    err = cov[0][0] + cov[1][1]*X*X 
    err = np.sqrt(err)
    return err


mu    = recta(0.5)
Sigma = bandaError(0.5)

print(f"y(x=0.5) = {round(recta(0.5),3)}+-{round(bandaError(0.5),3)}")

plt.errorbar(x,y, yerr = sigma, marker='o', linestyle='none')
plt.plot(np.linspace(0,5,11), recta(np.array(np.linspace(0,5,11))))
plt.fill_between(np.linspace(0,5,11), recta(np.array(np.linspace(0,5,11)))-bandaError(np.linspace(0,5,11)), recta(np.array(np.linspace(0,5,11)))+bandaError(np.linspace(0,5,11)), color = 'lightblue' )
plt.fill_between(np.linspace(0,5,11), recta(np.array(np.linspace(0,5,11)))-bandaErrorMala(np.linspace(0,5,11)), recta(np.array(np.linspace(0,5,11)))+bandaErrorMala(np.linspace(0,5,11)), color = 'lightgreen', alpha = 0.2 )
plt.xlim([0,5])
plt.ylim([1,6])
plt.grid()
plt.show()

# --------------------------------------------------------------------

def linear(x, m, b):
    return m*x + b

def MC():
    Y = []
    for i in x: Y.append( rnd.gauss(recta(i), sigma) )
    
    popt, pcov = curve_fit(linear, x, Y, sigma=[sigma for i in range(N)])  
    
    m = popt[0]
    b = popt[1]
    
    return linear(np.array(0.5), m, b)

l = []
for i in range(1000): l.append(MC())

nbins          = 30
hist, binEdges = np.histogram(l, density = True, bins = nbins)
fig, ax        = plt.subplots()
binWidth       = 0.1

yerr = np.sqrt( hist/(1000*binWidth) )
x_domain = np.linspace(min(binEdges), max(binEdges), 100)

ax.bar(binEdges[:-1]+(binEdges[1]-binEdges[0])/2, hist, width=binWidth, align='center', color='cornflowerblue', capsize=2, alpha=1, label='MC', yerr=yerr, ecolor='blue')
plt.plot(x_domain, stats.norm.pdf(x_domain, mu, Sigma))

plt.show()





