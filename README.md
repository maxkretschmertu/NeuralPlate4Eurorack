# NeuralPlate4Eurorack

Real-time nonlinear plate resonator based on modal synthesis, FEM-generated training data, and a neural network model.

The project combines:

- a coupled resonant filter bank for nonlinear modal energy transfer
- finite-element modal analysis of 2D plate geometries for reference dataset generation
- a neural network approximating the FEM modal solution to seamlessly morph between geometries
- real-time audio synthesis with strike and stereo pickup positions

The current implementation is a Python prototype intended as the basis for a future Eurorack implementation. It is heavily based on the filte rbank approach proposed by Poirot et al. (2023) (HAL-04154118).

To test the application, run filter_resonator_plate_nn_v15.py. To change pickup positions, use left- and rightclick. To change strike position, use shift + leftclick. 
For generating your own dataset and model, look into /neural and use generate_dataset.py and train.py. 

---

## Overview

The synthesis system models a vibrating plate as a bank of resonant modes.

For each mode, the system requires:

- a resonance frequency
- a damping coefficient
- mode gain at the excitation position
- mode gains at the output / pickup positions

Instead of solving the FEM problem in real time, a neural network is trained on precomputed FEM reference data and predicts the required modal parameters from the current plate geometry and strike/pickup positions.
In this implementation the NN does not predict frequencies directly but outputs a geometry-depended frequency factor, which is used to calculate frequencies depending on size and material parameters outside of the NN.

The resulting modes are processed by a coupled resonant filter bank. Nonlinear behaviour is introduced through energy redistribution between modes, based on the approach proposed by Poirot et al. (2023).

---

## System Architecture

```text
Plate geometry
(morph, aspect)
        |
        v
   PlateNet neural network
        |
        +----------------------+
        |                      |
        v                      v
Modal frequency factors    Mode Gains
       μ_k                  φ_k(x, y)
        |                      |
        v                      |
Physical frequency scaling    |
(D, rho, H, size)             |
        |                      |
        +----------+-----------+
                   |
                   v
          Modal filter bank
                   |
          damping + excitation
                   |
                   v
       nonlinear mode coupling
                   |
                   v
       left / right pickups
                   |
                   v
             stereo audio
