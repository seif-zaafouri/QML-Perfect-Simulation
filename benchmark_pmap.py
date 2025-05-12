import os
batch_size = os.cpu_count() - 4 # Adapt this parameter according to your machine. Careful, allocating full CPU leads to bluescreen on Windows.
# Pmap uses one CPU per circuit, so the number of CPU allocated is also equal to the batch size.
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={batch_size}" #We prepare the use of pmap
os.environ['LOKY_MAX_CPU_COUNT'] = f'{batch_size}' # To silence some warning messages


import pennylane as qml
import numpy as np

import jax
import jax.numpy as jnp

from sklearn.datasets import make_blobs
from sklearn import preprocessing
import matplotlib.pyplot as plt

from time import time

def fidelity_calculation(state):
    fidelity = jnp.abs(state[0]) ** 2
    return fidelity

p_fidelity_calculation = jax.pmap(fidelity_calculation)

def x1_fixed_circuit(n_qubits, n_features, x1):
    dev = qml.device("default.tensor", method='mps', wires=n_qubits)
    @jax.jit
    @qml.qnode(dev, interface="jax-jit")
    def circuit(x2):
        
        for i in range(n_qubits):
            qml.H(wires=i)
            qml.RZ(x1[i % n_features], wires=i)
            qml.RY(x1[i % n_features], wires=i)
            qml.RY(-x2[i % n_features], wires=i)
            qml.RZ(-x2[i % n_features], wires=i)
            qml.H(wires=i)
        
        state = qml.state()
        return state # The calculation of fidelity is done outside the circuit
    return circuit

def similarity_matrix_jax(X,n_features, n_qubits):
    n_samples = len(X)
    sim_matrix = jnp.diag(jnp.ones(n_samples))
    X_jax = jnp.array(X)
    
    timeL=[]
    # timeIf =[]
    # timeCirc = []
    timeSpace = []
    # timeFid = []
    
    for i in range(n_samples):
        #beg = time()
        
        circuit = x1_fixed_circuit(n_qubits, n_features, X_jax[i])
        circuit = jax.jit(circuit)
        p_circuit = jax.pmap(circuit)
        
        ## We run for X[:i], but we cut it in batch_size for use with pmap
        n_iter = i//batch_size + 1       
        
        #timeCirc.append(time() - beg)
        
        
        for j in range(n_iter): 
            ## Initialize batch
            
            #beginning = time()
            
            if j == n_iter - 1:
                endI = i+1
                
            else:
                endI = (j+1)*batch_size
                
            batch_parameters = jnp.array([X[i] for i in range(j*batch_size,endI)])
            
            ## Add time for if
            #durationIf = time() - beginning
            #timeIf.append(durationIf)
            
            ## Begin calculation
            
            time1 = time()
            
            states = p_circuit(batch_parameters)
            
            duration = time() - time1
            timeL.append(duration)
            
            ## End calculation
            
            ## We begin post processing
            
            ## We calculate the fidelity
            #beg=time()
            
            fidelity = p_fidelity_calculation(states) # To gain more time, we also parallelized the calculation of fidelity. This yielded very minor gains.
            
            #timeFid.append(time() - beg)
        
            ## Now we calculate the similarity matrix
            
            beg = time()
            
            # This corresponds to most of the run time. Attempts were made to simplify it, but without success.
            sim_matrix = sim_matrix.at[i, j*batch_size:endI].set(fidelity)
            sim_matrix = sim_matrix.at[j*batch_size:endI, i].set(fidelity.T)
            
            timeSpace.append(time() - beg)
            
            
    print(f'Total circuit time: {sum(timeL)}')
    # print(f'Total if time: {sum(timeIf)}')
    print(f'Total space time: {sum(timeSpace)}')
    # print(f'Total circuit time: {sum(timeCirc)}')
    # print(f'Total fidelity time: {sum(timeFid)}')
            
    return sim_matrix



## Testing.

NB_DATA = 50
NB_CLUSTER = 3

X, y = make_blobs(n_samples=NB_DATA, centers=NB_CLUSTER, n_features=2, random_state=3)
min_max_scaler = preprocessing.MinMaxScaler()
X = np.pi * min_max_scaler.fit_transform(X) # rescale data between between [-pi/2, pi/2]


from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering

#qubit_range = [2, 8, 24, 48, 60, 100] #A bit ambitious
qubit_range = [2, 8, 24] #A bit less ambitious

for nqubits in qubit_range:
    print(f"Number of qubits: {nqubits}")
    beginning = time()
    sim_matrix = similarity_matrix_jax(X,n_features=2, n_qubits=nqubits)
    print(f"Execution time {time() - beginning}")
    clustering = SpectralClustering(n_clusters=NB_CLUSTER, affinity='precomputed')
    labels_pred = clustering.fit_predict(sim_matrix)
    score = normalized_mutual_info_score(labels_pred, y)
    print(f"NMI Clustering Score: {score:.3f}") 
    
# Memory time has become the limiting factor.