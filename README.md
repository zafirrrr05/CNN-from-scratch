# CNN From Scratch – Educational Implementation

### Overview

This repository contains a minimal Convolutional Neural Network (CNN) implemented completely from scratch using low-level numerical operations.  
The project is designed for learning and understanding how convolutional neural networks work internally, rather than for performance or real-world deployment.

All major parts of the training pipeline are implemented manually, including forward propagation, loss computation, and backpropagation.

---

### Project Description

This project implements a small CNN for handwritten digit classification.  
Every layer is written explicitly, including the convolution operation and its gradient computation.

The implementation includes:

- a custom two-dimensional convolution layer with multiple filters  
- a ReLU activation layer  
- a flattening step that converts feature maps into vectors  
- a fully connected (dense) output layer  
- a softmax classifier  
- categorical cross-entropy loss  
- manual parameter updates using gradient descent  

The complete learning process is handled explicitly without using any deep-learning frameworks.

---

### Network Architecture

The network follows a simple and transparent structure:

- input image of size 1 × 28 × 28  
- one convolution layer with 8 filters of size 3 × 3 (valid convolution, no padding)  
- ReLU activation  
- flattening of the convolution feature maps  
- one fully connected layer that produces 10 class scores  
- softmax output layer  

This compact architecture is intentionally chosen to keep the internal operations easy to inspect and understand.

---

### Purpose

The main goal of this project is to help understand:

- how multi-channel 2D convolution is implemented from scratch  
- how gradients are computed for convolution kernels and inputs  
- how backpropagation flows through convolution, activation, and dense layers  
- how a complete CNN training pipeline can be built without deep-learning frameworks  

---

### Limitations

This implementation is intentionally minimal and has several limitations:

- training is performed one sample at a time  
- convolution uses only valid mode (no padding and no stride support)  
- no pooling layers are implemented  
- only a single convolution layer is used  
- no regularization methods are included  
- no advanced optimizers such as Momentum or Adam are implemented  

Because of these design choices, the implementation is slow and not suitable for production use.

---

### Intended Audience

This repository is intended for:

- students learning how convolutional neural networks work internally  
- anyone interested in understanding and experimenting with low-level CNN implementations  
- readers who want to study backpropagation for convolution layers in detail  

The project prioritizes clarity and educational value over efficiency and scalability.
