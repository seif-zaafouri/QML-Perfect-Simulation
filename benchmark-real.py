from qiskit.circuit.library import QFT
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
from time import time, sleep
import matplotlib.pyplot as plt
import numpy as np

## Benchmark parameters - to complete ##

IBM_TOKEN="<IBM_token_here>"
qubit_range = [2, 8, 24, 48, 60, 100]
PLOT = True # Set to true to see plotted results

## Qiskit circuits definition ##

def circuit_benchmark(n_qubits):
    qc = QFT(n_qubits)
    qc.measure_all()
    return qc

qc_list = [circuit_benchmark(i) for i in qubit_range]

## Qiskit simulation benchmark ##

print("** Launching Qiskit simulation **\n")

simulator = AerSimulator(method="matrix_product_state")

simulation_results = []
for index,circuit in enumerate(qc_list):
    circ = transpile(circuit, simulator)
    start = time()
    simulation_results.append(simulator.run(circ, shots=1024).result())
    delta = time() - start
    print(f"Job n°{index} | Nb qubits: {qubit_range[index]} | Simulation computation time: {delta:.6f}s")

if PLOT:
    print("\n** Plotting simulation results **")
    for sim_result in simulation_results:
        counts = sim_result.get_counts(circ)
        plot_result = plot_histogram(counts)
        plt.show()

## Qiskit circuit preparation and real execution ##

print("\n** Launching Qiskit real execution **\n")

service = QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q/open/main', token=IBM_TOKEN)
backend = service.least_busy(operational=True, simulator=False)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
qc_list = [pm.run(i) for i in qc_list]

sampler = Sampler(backend)

jobs_list = []
for index,circuit in enumerate(qc_list):
    job = sampler.run([circuit])
    jobs_list.append(job)
    print(f"Job n°{index} | Nb qubits: {qubit_range[index]} | sent successfully | " + 
          f"Estimated execution time: {job.usage_estimation['quantum_seconds']:.6f}s")

print("\n** Waiting for results **\n")
for index, job in enumerate(jobs_list):
    while not job.done():
        sleep(1)
    print(f"Job n°{index} | Nb qubits: {qubit_range[index]} | " +
          f"Estimated execution time: {job.usage_estimation['quantum_seconds']:.6f}s | Real execution time: {job.usage()}s")

real_results_list = [job.result() for job in jobs_list]

if PLOT:
    print("\n** Plotting real results **")
    for real_result in real_results_list:
        res = np.array(real_result[0].data.meas.get_bitstrings())
        values, counts = np.unique(res, return_counts=True)
        counts = {values[i] : counts[i] for i in range(len(values))}
        plot_result = plot_histogram(counts)
        plt.show()