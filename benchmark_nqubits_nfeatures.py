import os
#batch_size = os.cpu_count() - 4
batch_size = 14
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={15}" #We prepare the use of pmap
os.environ['LOKY_MAX_CPU_COUNT'] = '15' #We prepare the use of joblib
import pennylane as qml
import numpy as np

import jax
import jax.numpy as jnp

from sklearn.datasets import make_blobs
from sklearn import preprocessing
import matplotlib.pyplot as plt

from time import time
import time

from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

nb_shots = 1024 #1024 by default
sim = AerSimulator(shots=nb_shots)

pm = generate_preset_pass_manager(backend=sim, optimization_level=1) 

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def embbeding_circuit(nb_qubits, nb_feature, name="input"):
    """
        nb_qubits : nomber of qubits in the circuit
        nb_feature : nb_feature of data
        This function return the circuit of embedding
    """
    qc = QuantumCircuit(nb_qubits)
    x = ParameterVector(name, nb_feature)
    for i in range(nb_qubits):
        qc.h(i)
        qc.rz(x[i % nb_feature], i)
        qc.ry(x[i % nb_feature], i)
    return qc

def fidelity_adjoint_circuit(nb_qubits, nb_feature):
    """
        nb_qubits : nomber of qubits in the circuit
        nb_feature : nb_feature of data
        This function insert the data in the circuit for fidelity with adjoint method
    """
    qc1 = embbeding_circuit(nb_qubits, nb_feature, name="input1")
    qc2 = embbeding_circuit(nb_qubits, nb_feature, name="input2")

    # Here, keep inverse as a circuit (without .to_gate()) for proper parameter binding later
    qc_total = QuantumCircuit(nb_qubits)
    qc_total.compose(qc1, inplace=True)
    qc_total.compose(qc2.inverse(), inplace=True)
    qc_total.measure_all()

    return qc_total

def similarity_adjoint(nb_qubits, nb_feature, X, backend, pass_manager):
    n = len(X)

    circ = fidelity_adjoint_circuit(nb_qubits, nb_feature)
    circ_list = []
    for i in range(len(X)):
        for j in range(len(X)):
            # We keep only upper values
            if i > j:
                circ_iter = circ.copy()
                circ_iter = circ_iter.assign_parameters({"input1["+str(k)+"]":X[i][k] for k in range(nb_feature)})
                circ_iter = circ_iter.assign_parameters({"input2["+str(k)+"]":X[j][k] for k in range(nb_feature)})
                circ_list.append(circ_iter)
    circ_list = [pass_manager.run(i) for i in circ_list]

    sampler = Sampler(backend)
    job = sampler.run(circ_list)
    result = job.result()

    index = 0
    res = np.eye(n) # similarity matrix to be completed
    for i in range(len(X)):
        for j in range(len(X)):
            # We fill only upper values
            if i > j:
                bitstring = result[index].data.meas.get_bitstrings()
                nb_shots = result[index].metadata["shots"]
                fidelity = bitstring.count('0'*nb_qubits)/nb_shots
                res[i,j] = fidelity
                res[j,i] = fidelity
                index += 1

    return res





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
    
    # timeL=[]
    # timeIf =[]
    # timeCirc = []
    # timeSpace = []
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
            
            #time1 = time()
            
            states = p_circuit(batch_parameters)
            
            #duration = time() - time1
            #timeL.append(duration)
            
            ## End calculation
            
            ## We begin post processing
            
            ## We calculate the fidelity
            #beg=time()
            
            fidelity = p_fidelity_calculation(states)
            
            #timeFid.append(time() - beg)
        
            ## Now we calculate the similarity matrix
            
            #beg = time()
            
            sim_matrix = sim_matrix.at[i, j*batch_size:endI].set(fidelity)
            sim_matrix = sim_matrix.at[j*batch_size:endI, i].set(fidelity.T)
            
            #timeSpace.append(time() - beg)
            
            
    # print(f'Total circuit time: {sum(timeL)}')
    # print(f'Total if time: {sum(timeIf)}')
    # print(f'Total space time: {sum(timeSpace)}')
    # print(f'Total circuit time: {sum(timeCirc)}')
    # print(f'Total fidelity time: {sum(timeFid)}')
            
    return sim_matrix







NB_DATA = 100
NB_CLUSTER = 3
nf = 16
qubit_range = range(nf, 21, 1)  

# Generate synthetic dataset
X, y = make_blobs(n_samples=NB_DATA, centers=NB_CLUSTER, n_features=nf, random_state=3)
min_max_scaler = preprocessing.MinMaxScaler()
X = np.pi * min_max_scaler.fit_transform(X)

# Lists to store results
qubits_list = []
time_list_qiskit = []
time_list_pennylane = []
nmi_scores = []

# Iterate over different numbers of qubits
for n_qubits in qubit_range:
    start_time = time.time()
    
    # Compute similarity matrix
    #sim_matrix_final = similarity_matrix(X, n_features = 2, n_qubits=n_qubits) #(pennylane)
    sim_matrix_final = similarity_adjoint(n_qubits, nf, X, sim, pm) #(qiskit)
    
    # Spectral clustering
    clustering = SpectralClustering(n_clusters=NB_CLUSTER, affinity='precomputed', random_state=3)
    labels_pred = clustering.fit_predict(sim_matrix_final)

    # Compute execution time and NMI score
    elapsed_time = time.time() - start_time
    score = normalized_mutual_info_score(labels_pred, y)
    
    # Store results
    qubits_list.append(n_qubits)
    time_list_qiskit.append(elapsed_time)
    nmi_scores.append(score)

    print(f"Qubits: {n_qubits}, Time: {elapsed_time:.2f}s, NMI Score: {score:.3f}")



    
    # Compute similarity matrix
    sim_matrix_final = similarity_matrix_jax(X, n_features = nf, n_qubits=n_qubits) #(pennylane)
    #sim_matrix_final = similarity_adjoint(n_qubits, 4, X, sim, pm) #(qiskit)
    
    # Spectral clustering
    clustering = SpectralClustering(n_clusters=NB_CLUSTER, affinity='precomputed', random_state=3)
    labels_pred = clustering.fit_predict(sim_matrix_final)

    # Compute execution time and NMI score
    elapsed_time = time.time() - start_time
    score = normalized_mutual_info_score(labels_pred, y)
    
    # Store results
    time_list_pennylane.append(elapsed_time)
    nmi_scores.append(score)

    print(f"Qubits: {n_qubits}, Time: {elapsed_time:.2f}s, NMI Score: {score:.3f}")    

# Plot results
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Number of Qubits')
ax1.set_ylabel('Execution Time (s) - Qiskit', color=color)
ax1.plot(qubits_list, time_list_qiskit, '-o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(qubits_list)

# Annotate NMI scores for Qiskit
for i, n_qubits in enumerate(qubit_range):
    nmi_score = nmi_scores[i]
    ax1.annotate(f'{nmi_score:.2f}', (n_qubits, time_list_qiskit[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Execution Time (s) - Pennylane', color=color)
ax2.plot(qubits_list, time_list_pennylane, '-o', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Annotate NMI scores for Pennylane
for i, n_qubits in enumerate(qubit_range):
    nmi_score = nmi_scores[len(qubit_range) + i]
    ax2.annotate(f'{nmi_score:.2f}', (n_qubits, time_list_pennylane[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=color)


plt.title('Quantum Clustering Performance by Number of Qubits')
plt.show()






#################################################################################################






