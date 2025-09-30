---
layout: post
title: Error Mitigation and suppression in QC
date: 2024-02-12 15:09:00
description: Introduction to Error Mitigation and Error Suppression in Quantum Computing
tags: physics programming code
categories: Quantum-Computing
featured: true
---

# Introduction to Error Mitigation and Error Suppression in Quantum Computing

In the current era of quantum computing, noise and decoherence are significant challenges that affect the quality of information obtained from quantum computers. While error correction is the ultimate goal to completely negate the effects of noise, it requires a huge overhead of both quantum and classical resources, making it impractical in the current NISQ (Noisy Intermediate-Scale Quantum) era. Instead, we can focus on reducing the effects of noise using techniques called error mitigation and error suppression, which aim to reduce hardware errors with a much smaller overhead.

## Error Mitigation with Measurement Error Mitigation (M3)

Measurement errors over $$N$$ qubits can be treated classically and satisfy the equation:

$$ \vec{p}_{noisy} = A\vec{p}_{ideal} $$

where $$\vec{p}*{noisy}$$ is a vector of noisy probabilities returned by the quantum system, $$\vec{p}*{ideal}$$ is the probabilities in the absence of measurement errors, and $$A\_{row,col}$$ is the $$2^N \times 2^N$$ complete assignment matrix, where each element in $$A$$ is the probability of bit string $$col$$ being converted to bit string $$row$$ by the measurement-error process.

M3 is a scalable quantum measurement error mitigation package that works in a reduced subspace defined by the noisy input bitstrings that are to be corrected. It solves the linear equation:

$$ (\tilde{A})^{-1}\tilde{A}\vec{p}_{ideal} = (\tilde{A})^{-1}\vec{p}_{noisy} $$

This linear equation can often be solved trivially using LU decomposition with modest computing resources. However, if the number of unique bitstrings is large or memory constraints are tight, the problem can be solved in a matrix-free manner using preconditioned iterative linear solution methods like GMRES or BiCGSTAB.

## Error Suppression with Dynamic Decoupling (DD)

Dynamic decoupling (DD) is used to increase the lifetime of quantum information by effectively disconnecting the environment. It scans the circuit for idle periods of time and inserts a DD sequence of gates in those spots. These gates amount to the identity, so they do not alter the logical action of the circuit but have the effect of mitigating decoherence in those idle periods.

It is important to have an optimal decoupling sequence that does not destroy qubit coherence instead of preserving it. Applying an X-gate twice is equal to an identity gate, so there is no logical change to the circuit.

## Error Mitigation with Twirled Readout Error eXtinction (T-REX)

T-REX involves "twirling" of gates, where noise is viewed as a set of extra probabilistic gates on top of the perfect circuit implementation. Every time the circuit is executed, this noisy gate set is conjugated with a gate randomly chosen from a "twirling set", often a set of Pauli operators (Pauli twirling).

Pauli twirling inserts pairs of Pauli gates (I, X, Y, Z) before and after entangling gates such that the overall unitary is the same, but implemented differently. This turns coherent errors into stochastic errors, which can then be eliminated by sufficient averaging.

## Digital Zero Noise Extrapolation (ZNE)

Digital Zero Noise Extrapolation (ZNE) is a technique for mitigating errors without the need for additional quantum resources. A quantum program is altered to run at different effect levels of processor noise, and the result is extrapolated to an estimated value at a noiseless level.

There are two methods for scaling noise in ZNE:

1. Pulse stretching (analog): Applying the same pulse stretched along a larger amount of time for a circuit, increasing the effective noise.
2. Local/Global folding (digital): Compiling the input circuit with a larger number of gates. Each gate $$G$$ is replaced by $$G$$, $$G^\dagger$$, $$G$$. On a simulator, the circuit remains unchanged, but on a real device, the noise increases.

ZNE amplifies the scale factor of noise to estimate the measured value in the absence of noise. However, it is not guaranteed to mitigate all errors.

## Conclusion

Error mitigation and suppression techniques like M3, DD, T-REX, and ZNE are crucial for reducing the effects of noise in the current NISQ era of quantum computing. While not as comprehensive as error correction, these methods provide a more practical approach to improving the quality of information obtained from quantum computers. As research in this field progresses, we can expect further advancements in error mitigation and suppression techniques, bringing us closer to the goal of fault-tolerant quantum computing.
