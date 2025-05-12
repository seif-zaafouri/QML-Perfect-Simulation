import pennylane as qml
import jax
import jax.numpy as jnp

import numpy as np
from time import time
import os

## Error calculation

def incertitude_type(mesures, student_coeff):
    nb = len(mesures)
    moyenne = sum(mesures)/nb
    ecart_carre = [(i-moyenne)**2 for i in mesures]
    incertitude_type = (sum(ecart_carre)/(nb*(nb-1)))**(1/2)
    return moyenne, incertitude_type*student_coeff

## Define your circuits here

def circuit_benchmark_A(weights):
    for i in range(n_qubits-1):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i+1)
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(n_qubits-1))

def circuit_benchmark_B():
    qml.QFT(wires=range(n_qubits))
    return qml.expval(qml.PauliZ(n_qubits-1))

## QML circuit definition

circuit_chosen = circuit_benchmark_A # Specify here the circuit you want to benchmark

def circuit_dev(dev, jax_bool):
    weights = np.random.random(size=(n_qubits-1, 2))
    if jax_bool:
        weights = jnp.array(weights)
        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit():
            return circuit_chosen(weights)
    else:
        @qml.qnode(dev)
        def circuit():
            return circuit_chosen(weights)
    
    return circuit

# Machine stats

print('########## BENCHMARK FOR PENNYLANE ##########')
print(f"Number of devices used by jax: {len(jax.devices())}")
print(f"Number of CPU cores: {os.cpu_count()} \n")

# Parameters options for the benchmark

qubit_range_init = [2, 8, 16, 24, 48, 100]   # Range of qubits to make performance calculations
nb_meas = 10                                 # Number of measures - change student coeff accordingly
coeff_student = 2

devices_types = [{"value": "default.qubit", "activate": True, "max_qubits_index": 3},
                 {"value": "default.tensor", "activate": True, "max_qubits_index": None}]
method_types = [{"value": "tn", "activate": True, "max_qubits_index": None}, 
                 {"value": "mps", "activate": False, "max_qubits_index": 4}]
jax_options = [{"value": False, "activate": True, "max_qubits_index": None}, 
                 {"value": True, "activate": True, "max_qubits_index": 4}]

print('########## Starting Benchmark ########## \n')

times_results = {}
# Tensor vs Qubit computation method selection
for name in [i for i in devices_types if i["activate"]]:
    if name["value"] == "default.qubit":
        types = [{"value": "N/A", "activate": True, "max_qubits_index": None}]
    else:
        types = [i for i in method_types if i["activate"]]
    # In case of tensor, choice of tensor method - tn or mps
    for type in types:
        # Choice to use or not Jax
        for jax_value in [i for i in jax_options if i["activate"]]:
            qubits_indexs = [i["max_qubits_index"] for i in [name, type, jax_value] if i["max_qubits_index"]]
            if qubits_indexs != []:
                qubit_range = qubit_range_init[:min(qubits_indexs)]
            else:
                qubit_range = qubit_range_init   
            name_v = name["value"]
            type_v = type["value"]
            jax_value_v = jax_value["value"]
            print(f"## Name: {name_v} | Type: {type_v} | Jax: {jax_value_v} ##")

            # Iterations on qubits
            for n_qubits in qubit_range:
                print(f"## {n_qubits} Qubits ##")
                measures = []
                for meas in range(nb_meas):
                    print(f"--> Measure n°: {meas+1}")
                    
                    if name_v == "default.qubit":
                        dev = qml.device(name_v, wires=n_qubits)
                    else:
                        dev = qml.device(name_v, wires=n_qubits, method=type_v)

                    if jax_value_v:
                        circuit = circuit_dev(dev, True)

                        # Compiling the circuit with JAX
                        jit_circuit = jax.jit(circuit)

                        # JAX runs async, so .block_until_ready() 
                        # blocks until the computation is finished.
                        jit_circuit().block_until_ready()

                        # Second call with jit.
                        start = time()
                        jit_circuit().block_until_ready()
                        execution_time = time() - start

                        # Save measure
                        measures.append(execution_time)

                    else:
                        circuit = circuit_dev(dev, False)

                        start = time()
                        circuit()
                        execution_time = time() - start

                        # Save measure
                        measures.append(execution_time)

                result = incertitude_type(measures, coeff_student)
                print(f"Execution time: {result[0]:0.8f} seconds")
                times_results[f"{name_v}-{type_v}-JAX={jax_value_v}-NbQubits={n_qubits}"] = result

print(times_results)
