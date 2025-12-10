# QNN_Heggs_bosons

# Quantum Neural Networks for Higgs Boson Classification  
### Variational Quantum Circuits with Amplitude & Angle Encoding (Qulacs)

This repository contains a full benchmarking study of **Quantum Neural Networks (QNNs)** applied to **Higgs boson signal/background classification** using the UCI Higgs dataset.  
We train **Variational Quantum Circuits (VQCs)** for binary classification using both:

- **Amplitude Encoding (AENC)**  
- **Angle Encoding (ANGEC)** with RX/RZ feature loading  

All models are implemented using **Qulacs**, a fast CPU/GPU quantum simulator.

---

## 🧬 Project Motivation

The Higgs boson dataset contains collision-event features from simulated proton–proton interactions.  
Events are labeled:

- **0 → background** (non-Higgs processes)  
- **1 → signal** (Higgs boson decay events)

These events are noisy, correlated, and nonlinear, making them a strong testing ground for **quantum models**, which may capture nonclassical correlations efficiently.

This repo explores how different **quantum data encoding strategies** and **circuit depths** affect performance.

---

## ⚛️ Quantum Model Summary

### **Amplitude Encoding (AENC)**  
- Maps 8 real-valued features → normalized quantum state vector.  
- Extremely compact representation (log₂ dimension).  
- Con: training becomes difficult as qubit count increases (barren plateaus).  
- Works best with **few qubits + deep circuits**.

### **Angle Encoding (ANGEC)**  
- Features encoded via **RX** and **RZ** rotations.  
- Supported by nearly all hardware.  
- Robust and expressive with shallow depth.  
- Scales well with qubits.

---

## 🧱 Variational Circuit Architecture

Each VQC consists of:

1. **Encoding Layer**  
   - AENC → amplitude state  
   - ANGEC → RX(xᵢ), RZ(xᵢ) per qubit

2. **Variational Layers** (repeated `L` times)  
Ry(θᵢ), Rz(φᵢ) on each qubit
Ring of CNOT entanglers


3. **Measurement**  
- Expectation value ⟨Z₀⟩ → sigmoid → probability(signal)

---

## 📊 Dataset Setup

We use a clean reduced subset:

- **8,000 training samples**
- **1,000 validation**
- **1,000 test**

This provides stable evaluation and keeps quantum simulation feasible.

---

## 🚀 Training Pipeline

- Numerical gradient estimation  
- Adam optimizer  
- Early stopping (patience = 4 epochs)  
- Model checkpointing (`best_model.npy`)  
- Full history tracking (loss & accuracy curves)  
- Benchmark plots & confusion matrices

All training loops include tqdm progress bars.

---

## 🧪 Benchmark Configurations

We benchmark these models:

| Encoding | Qubits | Layers |
|----------|--------|--------|
| AENC     | 4, 8   | 2, 4, 6 |
| ANGEC    | 4, 8   | 2, 3, 6 |

All models trained on identical dataset splits.

---

## 🏆 Results Summary

### **Best amplitude model**
| Qubits | Layers | Accuracy |
|--------|--------|----------|
| **4** | **L6** | **0.723** |

### **Best angle-encoding model (overall winner)**  
| Qubits | Layers | Accuracy |
|--------|--------|----------|
| **4** | **L6** | **0.758** |

### Key Findings

- Angle encoding achieved **higher accuracy** and **smoother convergence**.  
- Increasing **layers** mattered more than increasing **qubits**.  
- 12-qubit amplitude models were too unstable to train (as expected).  
- QNNs can learn meaningful physical structure from Higgs features.

---

## 📈 Example Benchmark Plot

> Amplitude vs Angle Encoding — Test Accuracy vs Layers  
*(Generated in notebook)*

- Blue = amplitude  
- Green = angle  
- Circles = 4 qubits  
- Squares = 8 qubits  

Angle encoding consistently dominates amplitude encoding.

---

## 🔍 Confusion Matrices

Confusion matrices are provided for all best-performing models:

- Amplitude 4q-L6  
- Angle 4q-L6  
- Amplitude 8q-L4  
- Angle 8q-L4  

These can be reproduced via:

```python
show_confusion("angle", 4, 6)
show_confusion("amplitude", 4, 6)
