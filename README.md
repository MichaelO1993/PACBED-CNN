# PACBED-CNN

Infer thickness and mistilt from position-averaged convergent beam electron
diffraction (PACBED) patterns using a convolutional neural network (CNN).

Since training the required models requires significant resources, and a trained
model is rather large compared to the size of an image, the inference can be
deployed efficiently as a web service and queried from clients.

This repository contains three components:

* Python, Gatan Microscopy Suite (GMS) and web clients
* Inference and web service
* Training scripts

# Python client library installation

* Clone repository
* Change to `client` directory in the repository
* `pip install .`

## Installation for Gatan Microscopy Suite

The GMS / Digital Micrograph clients require [installation of the optional Python environment for GMS](https://www.gatan.com/python-installation). This installs Miniconda and creates a
dedicated Python environment in C:\ProgramData\Miniconda3\envs\GMS_VENV_PYTHON. The Python interpreter in GMS runs from this environment. Installing the web service client library in this environment makes it available in GMS Python.

Run "Anaconda Promt" as administrator and activate the GMS Python environment for installation:

```
conda activate C:\ProgramData\Miniconda3\envs\GMS_VENV_PYTHON
```

After that, install the client as described above into that environment.

# Examples for clients

The GMS clients and a Jupyter notebook client can be found in
`client/examples/`. For GMS, both a minimal example and a convenient GUI client
that combines acquisition and thickness inference are available. They are
DMScript files that can be opened and run in GMS.

The inference web service offers a web-based form that can be used from a
browser and includes a JavaScript client for the service. The template for the form
with the JavaScript client can be found at `webapi/src/pacbed_api/templates/form.html`

# Web service installation

To install and run the web service:

- Clone this repository
- Change into the `webapi` directory
- `pip install -e .` to install the package and required dependencies (change to an appropriate Python virtual environment before)
- Download and extract the example data (see below)
- Change into the extracted PACBED-CNN-data directory and run the server via `uvicorn pacbed_api:app --port 8230`
- Access the web GUI at http://localhost:8230/ or use the client library as mentioned above

# Example data

Data for testing, can be downloaded under:
https://cloud.tugraz.at/index.php/s/wJ4ZfWdsHK3p5WX

The data contains two systems (trained CNNs and simulated PACBEDs):
  - Rutile (ID-number: 0)
  - Strontium titanate (ID-number: 1)

Every material system gets its own ID, saved in the Register.csv including all system parameters.

# Simulation and Training

To add new systems:

- Copy *Simulation_PACBED.ipynb* and *Training_CNN.ipynb* to the folder, where your example data is located.
- *Simulation_PACBED.ipynb*
  - Modify simulation parameters to the requested system
  - Run the script
- *Training_CNN.ipynb*
  - Change ID to train the requested system
  - Run the script
