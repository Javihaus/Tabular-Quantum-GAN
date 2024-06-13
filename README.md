# Quantum Generative Adversarial Network for Synthetic Data Generation
![qcircuit](https://github.com/Javihaus/Tabular-Quantum-GAN/assets/23658379/aaeb41f3-fadb-414c-8ab3-a89bba42f0cd)
<div align="center">Image from https://pennylane.ai/qml/demos/tutorial_quantum_gans/</div>

## Overview

A Quantum Generative Adversarial Network (QGAN) implementation, specifically utilizing the circles dataset from Scikit-learn, is contained in this repository. The generator utilizes a quantum circuit to generate data that closely replicates the original dataset. This research investigates the convergence of quantum computing with machine learning, with the goal of utilizing the capabilities of quantum circuits for generative models. This project is based on a pennylane.ai example demo tutorial (https://pennylane.ai/qml/demos/tutorial_quantum_gans/)

## Features

- **Quantum Generator**: Utilizes a parameterized quantum circuit built with Pennylane to generate synthetic data.
- **Discriminator**: A classical neural network implemented in PyTorch to distinguish between real and synthetic data.
- **Training Integration**: Synthetic data generation is seamlessly integrated within the training loop, enhancing efficiency and performance.
- **Flexible Data Preprocessing**: Supports various preprocessing techniques to ensure the input data is well-normalized for optimal performance.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/qgan-synthetic-data.git
    cd qgan-synthetic-data
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the Data**:
   The circles dataset from Scikit-learn is used as the target data. The data is preprocessed and normalized to ensure effective training. In the code you can find also the dataset Swiss roll (Machine Learning: An Algorithmic Perspective”, 2nd edition, Chapter 6, 2014).

    ```python
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, noise=0.01, random_state=42)
    ```

2. **Define the Quantum Circuit**:
   A parameterized quantum circuit is defined using Pennylane. The circuit depth and the number of qubits can be adjusted to increase the complexity and capacity of the generator.
   
    ```python
    # Quantum variables
    n_qubits = 5  # Total number of qubits 
    n_a_qubits = 1  # Number of ancillary qubits 
    q_depth = 12  # Depth of the parameterised quantum circuit. Increasing circuit depth 
    n_generators = 4  # Number of subgenerators for the patch method
    ```
You can adjust the number of layers and types of quantum gates (marked as optional in the code) in your quantum generator circuit. Sometimes deeper circuits with more parameters can capture more complex distributions.
   ```python
    import pennylane as qml
    from pennylane import numpy as pnp

    @qml.qnode(dev, diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(q_depth, n_qubits, 3)

        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)

        for i in range(q_depth):
            for y in range(n_qubits):
                qml.RY(weights[i][y][0], wires=y)
                #qml.RX(weights[i][y][1], wires=y) #optional quantum gate
                #qml.RZ(weights[i][y][2], wires=y) #optional quantum gate

            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
            #qml.CNOT(wires=[n_qubits - 1, 0]) #optional

        return qml.probs(wires=list(range(n_qubits)))
    ```

3. **PatchQuantumGenerator**:
This generator is designed to generate data in patches rather than generating entire samples at once. This approach is commonly used in generative models to facilitate better training dynamics, especially when dealing with high-dimensional data or complex distributions.
A "patch" refers to a subset of the generated data. Instead of generating entire samples in one go, the generator generates smaller patches of data. Each patch represents a partial view of the data distribution. By generating data in patches, the generator can focus on capturing local patterns and structures within the data distribution, which can lead to more effective training and better overall performance.
The decision to use four generators is likely motivated by the desire to increase the diversity and richness of the generated data. Each generator is responsible for generating a distinct set of patches, allowing for a more comprehensive exploration of the data distribution.
By having multiple generators operating concurrently, the model can leverage diverse perspectives and strategies to capture different aspects of the data distribution. This can help prevent mode collapse and improve the overall quality of the generated data.


4. **Training the QGAN**:
   The training loop integrates the generation of synthetic data within the forward pass of the generator. The discriminator and generator are trained iteratively, ensuring that the generator improves in producing realistic synthetic data.

    ```python
    # Training loop
    for epoch in range(num_epochs):
        for real_data in data_loader:
            real_data = real_data.to(device)

            # Train discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data)
            d_loss = discriminator_loss(real_output, fake_output)
            gp = gradient_penalty(discriminator, real_data, fake_data)
            d_loss += lambda_gp * gp
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)
            g_loss = generator_loss(fake_output)
            g_loss.backward()
            optimizer_g.step()
    ```

5. **Evaluating the Model**:
   Various metrics and visual inspection can be used to evaluate the performance of the QGAN and the quality of the generated data. In the code we use two metrics: Cluster measure and Propensity score mean-squared error or pmse (Woo M.-J., Reiter J. P., Oganian A., Karr A. F. Global Measures of Data Utility for Microdata Masked for Disclosure Limitation. J Priv Confidentiality. 2009;1(1):111–24)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bug fixes, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
This format uses Markdown to create headings, code blocks, and lists, making it suitable for a GitHub README.md file.




Add a point where I need to define parameters for the quantum circuit as nqubits
ChatGPT
Sure! Here's the updated README with an additional section for defining parameters for the quantum circuit, such as n_qubits:

markdown
Copiar código
# Quantum Generative Adversarial Network for Synthetic Data Generation

## Overview

This repository contains the implementation of a Quantum Generative Adversarial Network (QGAN) designed to generate synthetic data, specifically using the circles dataset from Scikit-learn. The generator leverages a quantum circuit to produce data that closely mimics the original dataset. This project explores the intersection of quantum computing and machine learning, aiming to harness the power of quantum circuits for generative models.

## Features

- **Quantum Generator**: Utilizes a parameterized quantum circuit built with Pennylane to generate synthetic data.
- **Discriminator**: A classical neural network implemented in PyTorch to distinguish between real and synthetic data.
- **Training Integration**: Synthetic data generation is seamlessly integrated within the training loop, enhancing efficiency and performance.
- **Advanced Loss Functions**: Incorporates Wasserstein loss and gradient penalty to stabilize the GAN training process.
- **Flexible Data Preprocessing**: Supports various preprocessing techniques to ensure the input data is well-normalized for optimal performance.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/qgan-synthetic-data.git
    cd qgan-synthetic-data
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the Data**:
   The circles dataset from Scikit-learn is used as the target data. The data is preprocessed and normalized to ensure effective training.

    ```python
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, noise=0.01, random_state=42)
    ```

2. **Define Parameters**:
   Define the parameters for the quantum circuit, such as the number of qubits (`n_qubits`), depth of the circuit (`q_depth`), and other relevant hyperparameters.

    ```python
    n_qubits = 4   # Number of qubits
    q_depth = 3    # Depth of the quantum circuit
    noise_dim = n_qubits  # Dimension of the noise input
    ```

3. **Define the Quantum Circuit**:
   A parameterized quantum circuit is defined using Pennylane. The circuit depth and the number of qubits can be adjusted to increase the complexity and capacity of the generator.

    ```python
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(q_depth, n_qubits, 3)

        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)

        for i in range(q_depth):
            for y in range(n_qubits):
                qml.RY(weights[i][y][0], wires=y)
                qml.RX(weights[i][y][1], wires=y)
                qml.RZ(weights[i][y][2], wires=y)

            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        return qml.probs(wires=list(range(n_qubits)))
    ```

4. **Training the QGAN**:
   The training loop integrates the generation of synthetic data within the forward pass of the generator. The discriminator and generator are trained iteratively, ensuring that the generator improves in producing realistic synthetic data.

    ```python
    import torch

    # Training loop
    for epoch in range(num_epochs):
        for real_data in data_loader:
            real_data = real_data.to(device)

            # Train discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data)
            d_loss = discriminator_loss(real_output, fake_output)
            gp = gradient_penalty(discriminator, real_data, fake_data)
            d_loss += lambda_gp * gp
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)
            g_loss = generator_loss(fake_output)
            g_loss.backward()
            optimizer_g.step()
    ```

5. **Evaluating the Model**:
   Various metrics and visual inspection can be used to evaluate the performance of the QGAN and the quality of the generated data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bug fixes, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
