# INPUT

# Select measured PACBED
file_PACBED = 'D:/Github/Measured_PACBED/PACBED_0.dm4'

# Select folder with CNN models and labels (for a specific system)
path_models = 'D:/Github/Trained_Models/'


# Optional validation of the predictions
validation_pred = 1



from ncempy.io import dm
import numpy as np
import glob
import os
from skimage import filters
from skimage.measure import regionprops
import tensorflow as tf
import pandas as pd

# Code Snippet limits GPU memory growth -> without, errors occur (may not necessary for cluster/other computers)
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# Load PACBED (dm-file)
img_PACBED = dm.dmReader(file_PACBED)
PACBED_measured = img_PACBED['data']

# Prepare measured PACBED

# Convert to grayscale, if loaded PACBED is RGB
if len(PACBED_measured.shape) > 2:
    PACBED_measured = PACBED_measured[:,:,0]


# Function for finding center of mass
def center_of_mass(image):
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    com = properties[0].centroid
    return com


# Function for cropping the maximum square around the center of mass (or shifting better? Square --> input for CNNs)
def cropping_center_of_mass(PACBED, com):
    # Finding maximum boundaries for the square
    if PACBED.shape[0] / 2 < com[0]:
        side_0 = PACBED.shape[0] - com[0]
    else:
        side_0 = com[0]
    if PACBED.shape[1] < com[1]:
        side_1 = PACBED.shape[1] - com[1]
    else:
        side_1 = com[1]

    square_side = side_0 if side_0 <= side_1 else side_1

    # Define boundaries
    x_0 = int(com[0] - square_side)
    x_1 = int(com[0] + square_side)
    y_0 = int(com[1] - square_side)
    y_1 = int(com[1] + square_side)

    # Crop image
    PACBED_cropped = PACBED[x_0:x_1, y_0:y_1]

    return PACBED_cropped


# Calculate Center of Mass
com = center_of_mass(PACBED_measured)
com_cropping = np.round(com)

# Crop measured PACBED
PACBED_cropped = cropping_center_of_mass(PACBED_measured, com_cropping)


# Get all filenames of CNNs and labels
model_names = [os.path.basename(x) for x in glob.glob(path_models + '*.h5')]
labels_names = [os.path.basename(x) for x in glob.glob(path_models + '*_labels.csv')]

# Load all models and labels
for i in range(0, len(model_names)):
    # Load all models to the correct variables
    if model_names[i].find('Scale') > -1:
        model_scale = tf.keras.models.load_model(os.path.join(path_models, model_names[i]))
    elif model_names[i].find('Thickness') > -1:
        model_thickness = tf.keras.models.load_model(os.path.join(path_models, model_names[i]))
    elif model_names[i].find('Mistilt') > -1:
        model_tilt = tf.keras.models.load_model(os.path.join(path_models, model_names[i]))

    # Load all labels to the correct variables
    if labels_names[i].find('Scale') > -1:
        label_scale = pd.read_csv(os.path.join(path_models, labels_names[i]))
    elif labels_names[i].find('Thickness') > -1:
        label_thickness = pd.read_csv(os.path.join(path_models, labels_names[i]))
    elif labels_names[i].find('Mistilt') > -1:
        label_mistilt = pd.read_csv(os.path.join(path_models, labels_names[i]))


# Extract required dimension from the thickness CNN (assumed that all CNNs have same input)
dim = model_thickness.layers[0].input_shape[0][1:]


# Make a grayscale or RGB image of the measured PACBED, depending on the CNN
if dim[2] == 3 and len(PACBED_cropped.shape) == 2:
    img = np.stack((PACBED_cropped,) * 3, axis=-1)
elif dim[2] == 1 and len(PACBED_cropped.shape) == 3:
    img = PACBED_cropped[:, :, 0]
else:
    img = PACBED_cropped
    
    
# Prepare image for CNN input by resizing and normalizing
img_arr = tf.keras.preprocessing.image.smart_resize(img, dim[0:2], interpolation='bilinear')
img_arr = img_arr/np.amax(img_arr)


# Make CNN-predictions

# Iterative scaling of the image
k = 0
while True:
    # Make scale prediction
    scale_prediction = model_scale.predict(img_arr[np.newaxis, :, :, :])  # call instead of predict may be faster
    # Get scaling value with the highest predicted value
    scale_pred = label_scale['Scale / []'][np.argmax(scale_prediction)]
    
    # Break loop if scaling is 5, maximum runs of the loop is exceeded or the prediction is too low
    if scale_pred == 1 or k > 5 or np.amax(scale_prediction) < 0.3:
        break
    else:
        # Scale image (with full pixels)
        img = tf.keras.preprocessing.image.apply_affine_transform(img, zx=1/scale_pred, zy=1/scale_pred, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0., order=1)
        # Resize and normalize image for next predictions
        img_arr = tf.keras.preprocessing.image.smart_resize(img, dim[0:2], interpolation='bilinear')
        img_arr = img_arr/np.amax(img_arr)
        # Loop counter
        k += 1


# Predict Thickness

# Make thickness prediction
thickness_prediction = model_thickness.predict(img_arr[np.newaxis, :, :, :])
# Get thickness value with the highest predicted value
thickness_pred = label_thickness['Thickness / A'][np.argmax(thickness_prediction)]

# Predict Mistilt

# Make mistilt prediction
mistilt_prediction = model_tilt.predict(img_arr[np.newaxis, :, :, :])
# Get mistilt value with the highest predicted value
mistilt_pred = label_mistilt['Mistilt / mrad'][np.argmax(mistilt_prediction)]



# OUTPUT

print('Predicted Thickness: ' + str(thickness_pred) + ' A')
print('Predicted Mistilt: ' + str(mistilt_pred) + ' mrad')




# VALIDATION

if validation_pred == 1:
    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Get grayscale scaled image to compare to the grayscale imported PACBED
    if len(img_arr.shape) > 2:
        img_arr_plot = img_arr[:,:,0]
    else:
        img_arr_plot = img_arr
    
    
    # Plot imported PACBED and the scaled PACBED, which is used for thickness and mistilt prediction
    fig1, (ax11, ax12) = plt.subplots(1, 2)
    fig1.suptitle('PACBED')
    ax11.imshow(PACBED_measured)
    ax11.set_title('Loaded PACBED')
    ax12.imshow(img_arr_plot)
    ax12.set_title('PACBED for prediction')
    fig1.tight_layout() 
    
    
    # Plot output of the thickness model and the mistilt model
    
    # Sort thickness values (if labels are not ascended ordered)
    thickness_sort_ind = np.argsort(label_thickness.iloc[:,0])
    thickness_pred_sorted = thickness_prediction[0,thickness_sort_ind]
    thickness_values_sorted = np.array(label_thickness.iloc[:,0][thickness_sort_ind])
    
    # Sort mistilt values (if labels are not ascended ordered)
    mistilt_sort_ind = np.argsort(label_mistilt.iloc[:,0])
    mistilt_pred_sorted = mistilt_prediction[0,mistilt_sort_ind]
    mistilt_values_sorted = np.array(label_mistilt.iloc[:,0][mistilt_sort_ind])

    fig2, (ax21, ax22) = plt.subplots(2, 1)
    # Plot output of thickness prediction
    ax21.plot(thickness_values_sorted/10,thickness_pred_sorted)
    ax21.set_title('Thickness Prediction')
    ax21.set_xlim([np.amin(thickness_values_sorted)/10, np.amax(thickness_values_sorted)/10])
    ax21.set_ylim([0, 1])
    ax21.set_xlabel('Thickness / nm')
    # Plot output of mistilt prediction
    ax22.plot(mistilt_values_sorted,mistilt_pred_sorted)
    ax22.set_title('Mistilt Prediction')
    ax22.set_xlim([np.amin(mistilt_values_sorted), np.amax(mistilt_values_sorted)])
    ax22.set_ylim([0, 1])
    ax22.set_xlabel('Mistilt / mrad')
    fig2.tight_layout()  



    # Plot simulated PACBEDs with predicted values
    
    # Location of the simulated PACBEDs required
    dataframe_path = 'D:/Github/PACBED-CNN/Trained_Models/df.csv'
    
    # Load dataframe (csv-file with out index)
    dataframe = pd.read_csv(dataframe_path)
    
    # Get specific convergence angle for plotting (otherwise to many plots)
    conv_angle_unique = np.unique(dataframe['Conv_Angle'])
    # Take middle convergence angle (if convergence angle is known, closest value can be taken)
    conv_angle_plot = conv_angle_unique[len(conv_angle_unique)//2]
    
    # Filter dataframe for the predicted values (open value is azimuth)
    filteredDataframe = dataframe[(dataframe['Thickness'] == thickness_pred) & (dataframe['Mistilt'] == mistilt_pred) & (dataframe['Conv_Angle'] == conv_angle_plot) ]
    filteredDataframe = filteredDataframe.reset_index(drop=True)
    
    # Plot simulated PACBEDs with different azimuth angle with the measured scaled PACBED
    ax3=[]
    fig3=plt.figure()
    
    for i in range(0, len(filteredDataframe)+2):
        # Add subplot
        ax3.append(fig3.add_subplot(np.ceil((len(filteredDataframe)+2)/4), 4, i+1))
        
        # Plot measured scaled PACBED last
        if i == len(filteredDataframe):
            subplot_title=('Measured PACBED')
            ax3[-1].set_title(subplot_title, fontsize=8)  
    
            # Plot image
            plt.imshow(img_arr_plot)
            ax3[-1].axis('off')    
            
        # Plot predicted values
        elif i == len(filteredDataframe) + 1:
            ax3[-1].set_axis_off()
            ax3[-1].text(0.1, 0.8, 'Parameters for simulated PACBEDs:', fontsize=8,weight='bold')
            ax3[-1].text(0.2, 0.6, 'Thickness: ' + str(thickness_pred/10) + ' nm', fontsize=8)
            ax3[-1].text(0.2, 0.4, 'Mistilt: ' + str(mistilt_pred) + ' mrad', fontsize=8)
            ax3[-1].text(0.2, 0.2, 'Conv. angle: ' + str(conv_angle_plot) + ' mrad', fontsize=8)
            
        # Plot simulated PACBEDs
        else:
            subplot_title=('Azimuth ' + str(filteredDataframe['Azimuth'][i]) + ' mrad')
            ax3[-1].set_title(subplot_title, fontsize=8)  
            
            # Load image
            img_sim = Image.open(filteredDataframe['Path'][i])
            img_sim_arr = np.array(img_sim)
            if len(img_sim_arr.shape) == 2:
                img_sim_arr = img_sim_arr[:,:,np.newaxis]
            img_sim_arr =tf.image.central_crop(img_sim_arr, central_fraction = 0.5)
            
            img_arr_sim_plot = img_sim_arr[:,:,0]
    
            # Plot image
            plt.imshow(img_arr_sim_plot)
            ax3[-1].axis('off')
    plt.show()

    