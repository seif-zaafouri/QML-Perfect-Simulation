import os 
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}" # To allocate more CPUs to the jax process. Note that without the use of pmap, not all CPUs are used.
os.environ['LOKY_MAX_CPU_COUNT'] = f'{os.cpu_count()}' # To silence some warning messages
import numpy as np
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from pennylane import numpy as np

import jax
import jax.numpy as jnp

from sklearn.datasets import make_blobs
from sklearn import preprocessing
import matplotlib.pyplot as plt

from time import time





print('########## JAX JIT BENCHMARK ##########')
print(f"Number of devices used by jax: {len(jax.devices())}")
print(f"Number of CPU cores: {os.cpu_count()}")
jax.config.update("jax_enable_x64", True) # To remove some warnings
jax.config.update("jax_disable_jit", False)  # Keep JIT enabled

def embedding_circuit(nb_qubits, nb_feature, dataPoint):
    """
        nb_qubits : nomber of qubits in the circuit
        nb_feature : nb_feature of data
        This function return the circuit of embedding
    """
    for i in range(nb_qubits):
        qml.H(wires=i)
        qml.RZ(dataPoint[i % nb_feature], wires=i)
        qml.RY(dataPoint[i % nb_feature], wires=i)
    

def inverse_embedding_circuit(nb_qubits, nb_feature, dataPoint):
    """
        nb_qubits : nomber of qubits in the circuit
        nb_feature : nb_feature of data
        This function return the circuit of embedding
    """
    for i in range(nb_qubits):
        qml.RY(-dataPoint[i % nb_feature], wires=i)
        qml.RZ(-dataPoint[i % nb_feature], wires=i)
        qml.H(wires=i)


def fidelity_adjoint_circuit_pennylane(nb_qubits, nb_feature, x1, x2):
    """
        nb_qubits : nomber of qubits in the circuit
        nb_feature : nb_feature of data
        This function insert the data in the circuit for fidelity with adjoint method
    """


    # Here, keep inverse as a circuit (without .to_gate()) for proper parameter binding later
    
    embedding_circuit(nb_qubits, nb_feature, dataPoint=x1)
    inverse_embedding_circuit(nb_qubits, nb_feature, dataPoint=x2)
    
    
    # the measure has to be incorporated at a later point 
    # qc_total.measure_all()
    
def x1_fixed_circuit(n_qubits, n_features, x1):
    dev = qml.device("default.tensor", method='mps', wires=n_qubits)
    @jax.jit
    @qml.qnode(dev, interface="jax-jit")
    def circuit(x2): #Note that in this case, the circuit only takes one dynamic parameter: x1 is fixed.
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



def similarity_matrix_jax(X,n_features, n_qubits): # Version with one less dynamic parameter
    n_samples = len(X)
    sim_matrix = np.eye(n_samples)
    X_jax = jnp.array(X)
    
    for i in range(n_samples):
        circuit = x1_fixed_circuit(n_qubits, n_features, X_jax[i])
        circuit = jax.jit(circuit) # Extra cost: we run the compilation of the circuit for each i. But, we now only have one dynamic parameter.
        for j in range(i):
            state = circuit(X_jax[j])
            fidelity = np.abs(state[0]) ** 2 # fidelity is the square of the amplitude of the |000> state
            sim_matrix[i, j] = fidelity
            sim_matrix[j, i] = fidelity
    return sim_matrix

def similarity_matrix(X,n_features, n_qubits): # Default version
    n_samples = len(X)
    sim_matrix = np.eye(n_samples)

    X_jax = jnp.array(X)
    
    
    dev = qml.device("default.tensor", method='mps', wires=n_qubits)
    @qml.qnode(dev)
    def circuit(x1, x2):
        fidelity_adjoint_circuit_pennylane(n_qubits, n_features, x1, x2)
        state = qml.state()
        return state # The calculation of fidelity is done outside the circuit
    
    for i in range(n_samples):
        for j in range(i):
            state = circuit(X_jax[i], X_jax[j]).block_until_ready()
            fidelity = np.abs(state[0]) ** 2 # fidelity is the square of the amplitude of the |000> state
            sim_matrix[i, j] = fidelity
            sim_matrix[j, i] = fidelity
    return sim_matrix


NB_DATA = 100
NB_CLUSTER = 3

X, y = make_blobs(n_samples=NB_DATA, centers=NB_CLUSTER, n_features=2, random_state=3)
min_max_scaler = preprocessing.MinMaxScaler()
X = np.pi * min_max_scaler.fit_transform(X) # rescale data between between [-pi/2, pi/2]


from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering

#qubit_range = [2, 8, 24, 48, 60, 100] #A bit ambitious
qubit_range = [2, 8, 10] #A bit less ambitious

for nqubits in qubit_range:
    print(f"**************Number of qubits: {nqubits}")
    print(f"##### 2 dynamic parameters #####")
    beginning = time()
    sim_matrix = similarity_matrix(X,n_features=2, n_qubits=nqubits)
    print(f"Execution time with 2 dynamic parameters {time() - beginning}")
    clustering = SpectralClustering(n_clusters=NB_CLUSTER, affinity='precomputed')
    labels_pred = clustering.fit_predict(sim_matrix)
    score = normalized_mutual_info_score(labels_pred, y)
    print(f"NMI Clustering Score: {score:.3f}") 
    
    
    print(f"##### 1 dynamic parameters #####")
    beginning = time()
    sim_matrix = similarity_matrix_jax(X,n_features=2, n_qubits=nqubits)
    print(f"Execution time with 1 dynamic parameters {time() - beginning}")
    clustering = SpectralClustering(n_clusters=NB_CLUSTER, affinity='precomputed')
    labels_pred = clustering.fit_predict(sim_matrix)
    score = normalized_mutual_info_score(labels_pred, y)
    print(f"NMI Clustering Score: {score:.3f}") 
    



