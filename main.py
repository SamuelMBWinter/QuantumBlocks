from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit import transpile
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import QFT, DraperQFTAdder, IntegerComparatorGate

# ---------------------

from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

# ---------------------

def qubit_list(*registers) -> list[Qubit]:
    register_list = [list(register) for register in registers]
    qubit_list = sum(register_list, [])
    return qubit_list

def QFTGate(num_qubits: int, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name=None) -> Gate:
    qc = QFT(num_qubits, approximation_degree, do_swaps, inverse, insert_barriers, name)
    return qc.to_gate()

def AdderPhaseGate(target_reg_length: int, summand_reg_length: int) -> Gate:
    # circuit setup

    qc = QuantumCircuit()

    target_reg = QuantumRegister(target_reg_length)
    summand_reg = QuantumRegister(summand_reg_length)

    qc.add_register(target_reg)
    qc.add_register(summand_reg)

    # gate construction
    for i in range(target_reg_length):
        for j in range(min(i+1, summand_reg_length)):
            qc.cp(2*np.pi/(2**(1+i-j)), summand_reg[j], target_reg[i])

    return qc.to_gate()

def ClassicalAdderPhaseGate(target_reg_length: int, k: int) -> Gate:
    # circuit setup
    bit_string = bin(k)[-1:1:-1]
    bits = [int(bit) for bit in bit_string]

    qc = QuantumCircuit()
    target_reg = QuantumRegister(target_reg_length)
    qc.add_register(target_reg)

    # gate construction
    for i in range(target_reg_length):
        for j in range(min(i+1, len(bits))):
           if bits[j] == 1:
                qc.p(2*np.pi/(2**(1+i-j)), target_reg[i])

    return qc.to_gate()

def AdderGate(target_reg_length: int, summand_reg_length: int) -> Gate:
    # circuit setup
    qc = QuantumCircuit()
    target_reg = QuantumRegister(target_reg_length)
    summand_reg = QuantumRegister(summand_reg_length)

    qc.add_register(target_reg)
    qc.add_register(summand_reg)
    
    # gate prepartations
    qft_n = QFTGate(target_reg_length, do_swaps=False)
    add_phase_gate = AdderPhaseGate(target_reg_length, summand_reg_length)

    # gate construction
    qc.append(qft_n, target_reg)
    qc.append(add_phase_gate, qubit_list(target_reg, summand_reg))
    qc.append(qft_n.inverse(), target_reg)
    
    return qc.to_gate()

def ClassicalAdderGate(target_reg_length: int, k: int) -> Gate:
    qc = QuantumCircuit()
    target_reg = QuantumRegister(target_reg_length)
    qc.add_register(target_reg)
    
    # gate prepartations
    qft_n = QFTGate(target_reg_length, do_swaps=False)
    add_phase_gate = ClassicalAdderPhaseGate(target_reg_length, k)

    # gate construction
    qc.append(qft_n, target_reg)
    qc.append(add_phase_gate, qubit_list(target_reg))
    qc.append(qft_n.inverse(), target_reg)
    
    return qc.to_gate()


def main():

    sol_bits = 3
    sum_bits = 2

    # Circuit setup
    qc = QuantumCircuit()

    target_reg = QuantumRegister(sol_bits)
    summand_reg = QuantumRegister(sum_bits)

    qc.add_register(target_reg)
    qc.add_register(summand_reg)

    # State Preparation
    qc.h(summand_reg[0])

    # Circuit Construction

    adder_gate = AdderGate(sol_bits, sum_bits)
    add_1_gate = ClassicalAdderGate(sol_bits, 1)
    add_2_gate = ClassicalAdderGate(sol_bits, 2)
    qc.append(adder_gate, qubit_list(target_reg, summand_reg))
    qc.append(add_1_gate, qubit_list(target_reg))
    qc.append(add_2_gate, qubit_list(target_reg))

    qc.measure_all()

    # Circuit Visualistion
    qc.draw('mpl')
    plt.show()
    
    # Circuit Simulation
    simulator = AerSimulator()
    circuit = transpile(qc, simulator)
    circuit.draw('mpl')
    plt.show()
    job = simulator.run(circuit)

    ## Unpacking Results
    result_ideal = job.result()
    counts_ideal = result_ideal.get_counts()

    ## Processing Results
    print('Counts(ideal):', counts_ideal)
    plot_histogram(counts_ideal)
    plt.show()

if __name__=="__main__":
    main()
