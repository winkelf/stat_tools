import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(124)

#########################################################################################################################
#                                              FUNCTIONS 
#########################################################################################################################

'''
Aca estan las funciones (nada raro, solo el calculo de las likelihoods, la minimizacion de logL, generacion de toys y plotting)
'''

def poisson_likelihood(mu, n, s, b, log=True):

    n = np.asarray(n)
    s = np.asarray(s)
    b = np.asarray(b)

    # expected counts
    nu = mu * s + b

    # protect against zeros
    nu = np.clip(nu, 1e-12, None)

    # log-likelihood for independent Poisson bins
    logL = np.sum(n * np.log(nu) - nu )#- gammaln(n + 1)

    if log:
        return logL
    else:
        return np.exp(logL)

def fit_mu(n, s, b, mu_initial=1.0):

    # function to minimize
    def neg_logL(mu):
        mu = mu[0]
        # enforce mu >= 0
        if mu < 0:
            return np.inf
        return -poisson_likelihood(mu, n, s, b, log=True)

    # run minimization
    result = minimize(
        neg_logL,
        x0=[mu_initial],
        bounds=[(0, None)],       # mu >= 0
        method='L-BFGS-B'
    )

    return result

def generate_toys(mu_test, bg_events, sig_events, bins, xmin=0, xmax=10, N_toys=1000):

    # Template histograms
    b, bin_edges = np.histogram(bg_events,  bins=bins, range=(xmin, xmax))
    s, _         = np.histogram(sig_events, bins=bins, range=(xmin, xmax))

    # Compute expected counts for this test mu
    nu = mu_test * s + b

    # Protect against numerical issues
    nu = np.clip(nu, 1e-12, None)

    # Generate toys: each bin independently Poisson-fluctuated
    toys = np.random.poisson(lam=nu, size=(N_toys, len(nu)))

    return toys, s, b

def plotToy(toyn):
    toy = toys[toyn]
    # Recreate bin edges (same as histogram)
    bin_edges = np.linspace(0, 10, 31)   # 30 bins → 31 edges
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    plt.figure()
    plt.bar(bin_centers, toy, width=(10/30), alpha=0.5, label='Toy')
    plt.title("Toy bkg model")
    plt.xlabel("Observable x")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.legend()
    plt.savefig('toy.pdf')

def plotLLH(mu_test,sig_events,bg_events):
    fig = plt.figure()
    xAxis = np.linspace(0,10,101)
    yAxis = [poisson_likelihood(mu_test, len(np.concatenate([bg_events, sig_events])), len(sig_events), len(bg_events), log=True) for mu_test in xAxis]
    plt.plot(xAxis, yAxis)
    plt.xlabel("LL")
    plt.legend()
    plt.tight_layout()
    plt.savefig('LL.pdf')


#########################################################################################################################
#                                              Global parameters
#########################################################################################################################
'''
Definimos el numero de eventos a simular y los parametros de las distribuciones correspondientes

    B: Exp decreciente
    S: Gaussiana
'''

# Parameters
N_bg  = 20000   # number of background events
N_sig = 100    # number of signal (resonance) events

# Background shape: exponential
tau = 2.0

# Signal shape: Gaussian
mean  = 3.0
sigma = 0.3

# Histogram settings
xmin, xmax = 0, 10
bins       = 30

# Numero de toys
N_toys  = 100000
#########################################################################################################################
#                                              SIMULATING ASIMOV DATASET 
#########################################################################################################################
'''

Simulamos y ploteamos S+B y Bonly

'''

# Generate background events
u = np.random.random(N_bg)
bg_events = -tau * np.log(1 - u * (1 - np.exp(-(xmax - xmin)/tau))) + xmin

# Generate signal events (Gaussian)
sig_events = np.random.normal(mean, sigma, N_sig)
sig_events = sig_events[(sig_events >= xmin) & (sig_events <= xmax)]

# Plot histograms
plt.figure()
plt.hist(bg_events, bins=bins, range=(xmin, xmax),                                 histtype='stepfilled', alpha=0.5, label='Background')
plt.hist(sig_events, bins=bins, range=(xmin, xmax),                                histtype='stepfilled', alpha=0.7, label='Gaussian Resonance')
plt.hist(np.concatenate([bg_events, sig_events]),  bins=bins, range=(xmin, xmax),  histtype='step', linewidth=2, label='Total')

#print(plt.hist(bg_events, bins=bins, range=(xmin, xmax),                                 histtype='stepfilled', alpha=0.5, label='Background')[0])
#print(plt.hist(sig_events, bins=bins, range=(xmin, xmax),                                histtype='stepfilled', alpha=0.7, label='Gaussian Resonance')[0])

n_obs, bin_edges = np.histogram(np.concatenate([bg_events, sig_events]),  bins=bins, range=(xmin, xmax))
b, _             = np.histogram(bg_events,  bins=bins, range=(xmin, xmax))
s, _             = np.histogram(sig_events, bins=bins, range=(xmin, xmax))

plt.xlabel("Observable x")
plt.ylabel("Counts")
#plt.yscale('log') 
plt.title("Smoothly falling Background + Gaussian Resonance")
plt.legend()
plt.tight_layout()
plt.savefig('1_sb.pdf')
plt.close()

plt.figure()
plt.hist(bg_events, bins=bins, range=(xmin, xmax), histtype='stepfilled', alpha=0.5, label='Background')
plt.xlabel("Observable x")
plt.ylabel("Counts")
#plt.yscale('log') 
plt.title("Smoothly falling Background")
plt.legend()
plt.tight_layout()
plt.savefig('2_bonly.pdf')
plt.close()

#mu_test = 1   # test signal strength
#LL = poisson_likelihood(mu_test, n_obs, s, b, log=True) 
#print("log-likelihood =", LL)
#plotLLH(mu_test,sig_events,bg_events)

result = fit_mu(n_obs, s, b) 
mu_hat = result.x[0]
logL_at_mu_hat = -result.fun
print("Best-fit mu =", mu_hat)
print("log-likelihood at mu_hat =", logL_at_mu_hat)
#########################################################################################################################
#                                              SIMULATE MEASUREMENT 
#########################################################################################################################
'''

Simulamos una medicion, pero S y B tienen que ser los mismos de antes porque son una prediccion de la teoria! 

'''
N_bg_obs  = 20000   # number of background events
N_sig_obs = 100    # number of signal (resonance) events

# Generate background events
u = np.random.random(N_bg_obs)
bg_events_obs = -tau * np.log(1 - u * (1 - np.exp(-(xmax - xmin)/tau))) + xmin

# Generate signal events (Gaussian)
sig_events_obs = np.random.normal(mean, sigma, N_sig_obs)
sig_events_obs = sig_events_obs[(sig_events_obs >= xmin) & (sig_events_obs <= xmax)]

# Generate obs number of events in measurement
n_obs_obs, bin_edges = np.histogram(np.concatenate([bg_events_obs, sig_events_obs]),  bins=bins, range=(xmin, xmax))

#print(np.histogram(np.concatenate([bg_events_obs, sig_events_obs]),  bins=bins, range=(xmin, xmax))[0])

#########################################################################################################################
#                                              SIMULATE MEASUREMENT 
#########################################################################################################################

'''

Construimos -logL(n_obs, s, b) y lo minimizamos para obtener el μ de maxima verosimilitud

Es decir, buscamos el μ mas consistente con n_obs dados s y b

'''

result_obs   = fit_mu(n_obs_obs, s, b)  
mu_hat_obs   = result_obs.x[0]
logL_hat_obs = -result_obs.fun

with PdfPages('3_q_mu.pdf') as pdf:

    # Escaneamos valores de mu
    for v in np.linspace(1,2,40):
        mu_test = v

        # Para este dado valor de μ, calculamos q_obs(μ) = -2ln( L(μ) - L(μ_hat) )
        logL_mu_test = poisson_likelihood(mu_test, n_obs_obs, s, b, log=True)

        qobs = -999

        if mu_hat_obs > mu_test:
            qobs = 0.0
        else:
            qobs = -2.0 * (logL_mu_test - logL_hat_obs)

        # Generamos N histogramas sampleados de mis predicciones de S y B, dado un valor de μ 
        toys, _ , _ = generate_toys(
            mu_test,
            bg_events,
            sig_events,
            bins=bins,
            xmin=0,
            xmax=10,
            N_toys=N_toys
        )
       
        #plotToy(0)

        # Ahora de cada uno de los N histogramas, calculamos N valores de q para obtener la distribucion  
        q = []
        for t in toys:

            # Fit μ-hat for this toy
            result_toy   = fit_mu(t, s, b)
            mu_hat_toy   = result_toy.x[0]
            mu_hat_toy   = max(mu_hat_toy, 0)
            logL_hat_toy = -result_toy.fun

            # logL at fixed μ_test for the toy
            logL_mu_test_toy = poisson_likelihood(mu_test, t, s, b, log=True)

            if mu_hat_toy > mu_test:
                q_mu_toy = 0.0
            else:
                q_mu_toy = -2.0 * (logL_mu_test_toy - logL_hat_toy)

            q.append(q_mu_toy)

        # Para la distribucion de mu calculamos el pVal dado el qobs 
        q    = np.array(q)
        pval = np.mean(q > qobs)

        print(f"mu value = {mu_test}, pval = {pval}")

        plt.figure()
        plt.hist(q, bins=np.linspace(0,15,60), alpha=0.5, label=f'Distribution of q (mu = {round(v,3)}, $pval = ${pval})')
        plt.title("")
        plt.xlabel("q")
        plt.ylabel("Counts")
        plt.yscale("log")
        plt.xlim([0,15])
        plt.ylim([1,5e5])
        plt.tight_layout()

        # Plot a vertical line at x=2.5
        plt.axvline(x=qobs, color='r', linestyle='--', label='$q_{obs}$')

        plt.legend()
        pdf.savefig()

        

