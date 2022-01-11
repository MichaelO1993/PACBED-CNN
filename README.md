# PACBED-CNN
PACBED Identification by CNNs consists of two python scripts (Training + Prediction).

# Training
The CNN_Training.py trains a pretrained Xception model for thickness, mistilt and scale prediction. The scale CNN is used at the prediction for better matching of the simulated and the measured PACBED before the thickness and mistilt prediction. This should increase the accuracy of the thickness and mistilt prediction.

The **input** for this script is the **path and name** of a pandas dataframe (csv-file), which has to contain following columns:
  - **'Path'**:        Path and name of the simulated PACBED
  - **'Thickness'**:   Corresponding thickness in Angstrom
  - **'Mistilt'**:     Corresponding mistilt in mrad
  - **'Conv_Angle'**:  Corresponding convergence angle of the incident beam in mrad
  - **optional parameters**, which are not important for training or prediction but may be important for the user (electron energy, crystal, composition, orientation)

Further, it has to be specified, which predictions should be trained ('Thickness', 'Mistilt', 'Scale') and the (hyper-)parameters for the CNN-training (dimension of the CNN-input, batch size and epochs).

The **outputs** of the script are trained models (h5-file) and a corresponding pandas dataframe, which contains the labels for the prediction. These are saved in the same location as the dataframe.

An optional validation of the training can be done. The output of the datagenerator can be plotted to check, if the data augmentation is suitable for the simulated PACBEDs, and a confusion matrix (predicted vs true) can be plotted. The matrix can be used to see, if predicted values are far from the true value. (This would be more problematic as wrong predicted but close values.)

# Prediction
The CNN_Prediction.py predict the thickness and mistilt of a measured PACBED (dm4-format).

The **input** for this script is the path and name of the measured PACBED-file and the folder, which contains the CNN-models and labels for the system with the correct conditions (electron energy, crystal, orientation, ...).

The script centers the measured PACBED by finding the center of mass and scales it iterative with the scale CNN-model. After this, the thickness and mistilt are predicted with CNN-models.

The **output** are the predicted thickness value and predicted the mistilt value.

An optional validation of the prediction contains:
  - Preprocessing: Comparison of the original PACBED with the scaled, resized and normalized PACBED
  - Prediction output: Plotting the output of the thickness and mistilt model
  - Comparison of the scaled PACBED with the predicted simulated PACBED (for this, dataframe and the simulated PACBEDs are required)

# Trainings dataset

Trained models for rutile (TiO2) in 001-direction at 80 kV with a convergence angle of 20.8 mrad (+/- 1 mrad) and measured PACBEDs can be downloaded from the following links. The last few PACBEDs are taken at 300 kV instead of 80 kV.

Trained models: https://cloud.tugraz.at/index.php/s/oGdDdyWLoJfm447

Measured PACBEDs: 
 - https://cloud.tugraz.at/index.php/s/YQr9FNon4ErWX8o (h5-format)
 - https://cloud.tugraz.at/index.php/s/bTZ6coARLQamtWk (tensorflow lite format)

The trainings dataset for this system can be downloaded under: https://cloud.tugraz.at/index.php/s/xmCrRkcqESzBkgp

The zip-folder from the trainings dataset contains a python-script (Creating_Dataframe.py) to generate a dataframe from the trainings dataset, which is required for the CNN-training and the validation at the CNN-prediction. In the script, the path, where the dataset is saved, and the path/filename for saving the dataframe have to be changed.

# Web API prototype

To run the web API prototype:

- Change into the `webapi` directory
- `pip install -e .` to install the package and required dependencies (change to an appropriate Python virtual environment before)
- run the server: `uvicorn pacbed_api:app --reload`
- access the API docs at http://localhost:8000/docs/
