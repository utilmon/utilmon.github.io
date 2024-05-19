---
layout: post
title: Simulating Molecules with QC
date: 2024-03-12 15:09:00
description: Variational Quantum Eigensolver for Simulating Molecules
tags: physics programming code
categories: Quantum-Computing
featured: true
---

# Variational Quantum Eigensolver for Simulating Molecules

Quantum computing is an exciting field that has the potential to solve complex problems that are intractable for classical computers. One such problem is simulating molecules, which is crucial for drug discovery, materials science, and many other fields. In this article, we will explore the Variational Quantum Eigensolver (VQE) algorithm and how it can be used to simulate molecules using quantum computers.

## Introduction to VQE

The Variational Quantum Eigensolver is a hybrid quantum-classical algorithm that aims to find the ground state energy of a molecule. The algorithm consists of two main components:

1. A parameterized quantum circuit (ansatz) that prepares the trial wave function of the molecule.
2. A classical optimizer that varies the parameters of the ansatz to minimize the energy of the molecule.

The VQE algorithm iteratively optimizes the parameters of the ansatz until it converges to the ground state energy of the molecule.

## Simulating H2 Molecule

Let's start by simulating a simple molecule, H2, using the VQE algorithm. We will use the STO-3G basis set and the PySCF driver to perform the Hartree-Fock calculation. The Hartree-Fock method is a classical approximation that provides a good starting point for the VQE algorithm.

First, we define the molecule and run the Hartree-Fock calculation:

```python
from qiskit_nature.drivers import PySCFDriver

molecule = "H .0 .0 .0; H .0 .0 0.739"
driver = PySCFDriver(atom=molecule)
qmolecule = driver.run()
```

Next, we analyze the properties of the molecule:

```python
n_el = qmolecule.num_alpha + qmolecule.num_beta
n_mo = qmolecule.num_molecular_orbitals
n_so = 2 * qmolecule.num_molecular_orbitals
n_q = n_so
e_nn = qmolecule.nuclear_repulsion_energy

print("Number of electrons: {}".format(n_el))
print("Number of molecular orbitals: {}".format(n_mo))
print("Number of spin-orbitals: {}".format(n_so))
print("Number of qubits: {}".format(n_q))
print("Nuclear repulsion energy: {}".format(e_nn))
```

For the H2 molecule, we have 2 electrons, 2 molecular orbitals, 4 spin-orbitals, and 4 qubits. The nuclear repulsion energy is 0.7160720039512857 Hartree.

Now, we can set up the VQE algorithm. We choose the `TwoLocal` ansatz with linear entanglement, a single repetition, and final rotation layer. We use the COBYLA optimizer to minimize the energy.

```python
from qiskit.circuit.library import TwoLocal
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter

num_particles = (qmolecule.num_alpha, qmolecule.num_beta)
num_spin_orbitals = 2 * qmolecule.num_molecular_orbitals

init_state = HartreeFock(num_spin_orbitals, num_particles, QubitConverter(JordanWignerMapper()))

ansatz = TwoLocal(num_spin_orbitals, ['ry'], 'cx', reps=1, entanglement='linear', skip_final_rotation_layer=False)
ansatz.compose(init_state, front=True, inplace=True)

from qiskit.algorithms.optimizers import COBYLA
optimizer = COBYLA(maxiter=500)
```

Finally, we run the VQE algorithm and compare the results with the exact energy obtained from diagonalization:

```python
from qiskit.algorithms import VQE

backend = Aer.get_backend('statevector_simulator')
initial_point = [0.01] * ansatz.num_parameters

vqe = VQE(ansatz, optimizer, quantum_instance=backend, initial_point=initial_point)
result = vqe.compute_minimum_eigenvalue(qubit_op)

print("Exact energy: ", exact_energy)
print("VQE energy: ", result.optimal_value)
print("Error: ", (result.optimal_value - exact_energy) * 1000, "mHa")
```

The VQE algorithm converges to the exact energy within the chemical accuracy (1 kcal/mol ≈ 1.6 mHa). This demonstrates the power of the VQE algorithm in simulating simple molecules.

## Simulating LiH Molecule

Now, let's move on to a more complex molecule, LiH. We follow the same steps as before, but this time we use the `FreezeCoreTransformer` to reduce the number of qubits required for the simulation. The `FreezeCoreTransformer` eliminates the core orbitals that do not contribute significantly to the chemical bonding.

```python
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.transformers import FreezeCoreTransformer

molecule = 'Li 0.0 0.0 0.0; H 0.0 0.0 1.5474'
driver = PySCFDriver(atom=molecule)

freezeCoreTransformer = FreezeCoreTransformer(True)
problem = ElectronicStructureProblem(driver, q_molecule_transformers=[freezeCoreTransformer])
```

We also use the `ParityMapper` with `two_qubit_reduction=True` to further reduce the number of qubits. This mapping takes advantage of the symmetries in the Hamiltonian to eliminate qubits.

```python
from qiskit_nature.mappers.second_quantization import ParityMapper

mapper = ParityMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

num_particles = (problem.molecule_data_transformed.num_alpha, problem.molecule_data_transformed.num_beta)
qubit_op = converter.convert(main_op, num_particles=num_particles)
```

For the LiH molecule, we end up with 4 qubits after applying the `FreezeCoreTransformer` and `ParityMapper`. We use a custom ansatz with linear entanglement and two layers of rotation gates.

```python
num_qubits = qubit_op.num_qubits
qc = QuantumCircuit(num_qubits)

for i in range(num_qubits):
    qc.ry(Parameter(f'ry{i}_1'), i)

qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

for i in range(num_qubits):
    qc.ry(Parameter(f'ry{i}_2'), i)

ansatz = qc
ansatz.compose(init_state, front=True, inplace=True)
```

We run the VQE algorithm with the SLSQP optimizer and compare the results with the exact energy:

```python
optimizer = SLSQP(maxiter=1000)

vqe = VQE(ansatz, optimizer, quantum_instance=backend, initial_point=[0.01] * ansatz.num_parameters)
result = vqe.compute_minimum_eigenvalue(qubit_op)

print("Exact energy: ", exact_energy)
print("VQE energy: ", result.optimal_value)
print("Error: ", (result.optimal_value - exact_energy) * 1000, "mHa")
```

The VQE algorithm converges to the exact energy within the chemical accuracy, demonstrating its effectiveness in simulating more complex molecules.

## Conclusion

In this article, we explored the Variational Quantum Eigensolver algorithm and its application in simulating molecules using quantum computers. We demonstrated how to simulate the H2 and LiH molecules using the VQE algorithm and compared the results with the exact energies obtained from diagonalization.

The VQE algorithm is a promising approach for simulating molecules on near-term quantum computers. By leveraging the power of quantum computers and classical optimizers, the VQE algorithm can provide accurate results for complex molecules that are intractable for classical computers.

As quantum hardware and algorithms continue to improve, we can expect the VQE algorithm to become a valuable tool for drug discovery, materials science, and many other fields that rely on accurate molecular simulations.
