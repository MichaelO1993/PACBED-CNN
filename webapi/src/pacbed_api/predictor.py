import glob
import os
import io

# from ncempy.io import dm
import numpy as np
from skimage import filters
from skimage.measure import regionprops
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale

from scipy.ndimage import zoom

# Use GPU
gpu = False

if gpu:
    # Code Snippet limits GPU memory growth -> without, errors occur
    # (may not necessary for cluster/other computers)
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU used or found")


def background_subtraction(pacbed_img, border=0.1):
    # Subtract Background (improves prediction at larger thicknesses)
    pacbed_background = pacbed_img.copy()
    border_px_x = int(border * pacbed_background.shape[0])
    border_px_y = int(border * pacbed_background.shape[1])
    pacbed_background[border_px_x:-border_px_x, border_px_y:-border_px_y] = 0
    background = np.sum(pacbed_background)/(pacbed_background.size - border_px_x*border_px_y)

    # background = np.mean(pacbed_img) / 4 # The higher the noise the larger the background should be
    pacbed_background_sub = pacbed_img - background
    pacbed_background_sub[pacbed_background_sub <= 0] = 0

    print('Background subtracted')

    return pacbed_background_sub


# Function for finding center of mass
def center_of_mass(image):
    # Threshold for contributing to the center of mass
    threshold_value = filters.threshold_otsu(image)
    # Generate mask
    labeled_foreground = (image > threshold_value).astype(int)
    # Calculate center
    properties = regionprops(labeled_foreground, image)
    com = properties[0].centroid

    return com


# Function for cropping the maximum square around the center of mass
def cropping_center_of_mass(PACBED, com):
    # Finding maximum boundaries for the cropping square
    if PACBED.shape[0] / 2 < com[0]:
        side_0 = PACBED.shape[0] - com[0]
    else:
        side_0 = com[0]
    if PACBED.shape[1] < com[1]:
        side_1 = PACBED.shape[1] - com[1]
    else:
        side_1 = com[1]

    # Using smallest square
    square_side = side_0 if side_0 <= side_1 else side_1

    # Define boundary indices
    x_0 = int(com[0] - square_side)
    x_1 = int(com[0] + square_side)
    y_0 = int(com[1] - square_side)
    y_1 = int(com[1] + square_side)

    # Crop image
    PACBED_cropped = PACBED[x_0:x_1, y_0:y_1]

    return PACBED_cropped


def center_PACBED(pacbed_img):
    # Calculate Center of Mass
    com = center_of_mass(pacbed_img)
    com_cropping = np.round(com)

    # Crop measured PACBED
    pacbed_img = cropping_center_of_mass(pacbed_img, com_cropping)

    print('PACBED centered')

    return pacbed_img


# Convert PACBED to smaller dimension to increase speed
def redim_PACBED(pacbed_img, dim=(680, 680)):
    # Convert to PIL-framework
    PACBED_img = Image.fromarray(pacbed_img)

    # Resize
    PACBED_img = PACBED_img.resize(dim, resample=Image.NEAREST)

    # Normalize for CNN
    PACBED_arr = np.asarray(PACBED_img)
    pacbed_img = (
        2 * PACBED_arr - np.amin(PACBED_arr) / (np.amax(PACBED_arr) - np.amin(PACBED_arr)) - 1
    ).astype(np.float32)

    print(f'PACBED dimensions changed to {dim}')

    return pacbed_img


def show(pacbed_raw, pacbed_processed):
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(pacbed_raw)
    ax1.set_title('Measured PACBED')
    ax2.imshow(pacbed_processed)
    ax2.set_title('Processed PACBED')

    fig.tight_layout()


class Predictor:
    def __init__(self, parameters_prediction):
        # Declare variables
        self.id_system = parameters_prediction['id_system']
        self.id_model = parameters_prediction['id_model']
        self.conv_angle = parameters_prediction['conv_angle']

        self.conv_angle_norm = None
        self.dataframe = None
        self.dim = None
        # Scale CNN
        self.interpreter_scale = None
        self.scale_input_details = None
        self.scale_output_details = None
        # Thickness CNN
        self.interpreter_thickness = None
        self.thickness_input_details = None
        self.thickness_output_details = None
        # Mistilt CNN
        self.interpreter_tilt = None
        self.tilt_input_details = None
        self.tilt_output_details = None
        # Labels
        self.label_scale = None
        self.label_thickness = None
        self.label_mistilt = None

        # Get paths
        self.Path_models, self.Path_dataframe = self.get_path(self.id_system, self.id_model)

        # Load models and labels
        self.load_files()

    def get_path(self, id_system, id_model):
        df_system = pd.read_csv('./data/Register.csv', sep=';', index_col='id')
        print('Loaded system:')
        print(df_system.loc[[id_system]])  # double brackets to keep in the dataframe format
        path = df_system.loc[id_system]['path']
        path = path.replace('\\', '/')
        df_model = pd.read_csv(os.path.join(path, 'models', 'Register_models.csv'), sep=';',
                               index_col='id')
        print('Loaded model:')
        print(df_model.loc[[id_model]])

        Path_models = df_model.loc[id_model]['path'].replace('\\', '/')
        Path_dataframe = os.path.join(path, 'simulation', 'df.csv')

        return Path_models, Path_dataframe

    def load_files(self):
        # Get all filenames of models and labels
        model_names = [
            os.path.basename(x)
            for x in glob.glob(os.path.join(self.Path_models, '*.tflite'))
        ]
        labels_names = [
            os.path.basename(x)
            for x in glob.glob(os.path.join(self.Path_models, '*_labels.csv'))
        ]

        # Load all models to the correct variable
        for model_name in model_names:
            if model_name.find('Scale') > -1:
                # Tensorflow lite framework
                self.interpreter_scale = tf.lite.Interpreter(
                    model_path=os.path.join(self.Path_models, model_name)
                )
                self.interpreter_scale.allocate_tensors()
                self.scale_input_details = self.interpreter_scale.get_input_details()
                self.scale_output_details = self.interpreter_scale.get_output_details()
            elif model_name.find('Thickness') > -1:
                # Tensorflow lite framework
                self.interpreter_thickness = tf.lite.Interpreter(
                    model_path=os.path.join(self.Path_models, model_name)
                )
                self.interpreter_thickness.allocate_tensors()
                self.thickness_input_details = self.interpreter_thickness.get_input_details()
                self.thickness_output_details = self.interpreter_thickness.get_output_details()
            elif model_name.find('Mistilt') > -1:
                # Tensorflow lite framework
                self.interpreter_tilt = tf.lite.Interpreter(
                    model_path=os.path.join(self.Path_models, model_name)
                )
                self.interpreter_tilt.allocate_tensors()
                self.tilt_input_details = self.interpreter_tilt.get_input_details()
                self.tilt_output_details = self.interpreter_tilt.get_output_details()

                # Extract required input dimension from the thickness CNN
        self.dim = self.thickness_input_details[1]['shape'][1:]

        # Load all models to the correct variable
        for label_name in labels_names:
            if label_name.find('Scale') > -1:
                self.label_scale = pd.read_csv(
                    os.path.join(self.Path_models, label_name), sep=';'
                )
            elif label_name.find('Thickness') > -1:
                self.label_thickness = pd.read_csv(
                    os.path.join(self.Path_models, label_name), sep=';'
                )
            elif label_name.find('Mistilt') > -1:
                self.label_mistilt = pd.read_csv(
                    os.path.join(self.Path_models, label_name), sep=';'
                )

        # Load dataframe (csv-file with out index)
        self.dataframe = pd.read_csv(self.Path_dataframe, sep=';')
        self.conv_angle_norm = self.get_conv_angle_norm()

        print('Models and Labels loaded.')

    def input_tensor_idx(self, input_details):
        if len(input_details[0]['shape']) == 4:
            idx_pacbed = 0
            idx_conv = 1
        else:
            idx_pacbed = 1
            idx_conv = 0

        return idx_pacbed, idx_conv

    def get_conv_angle_norm(self):
        # Used convergence angle
        conv_angle_unique = np.unique(self.dataframe['Conv_Angle'])
        # Calculate scaled convergence angle
        conv_angle_norm = ((self.conv_angle - np.amin(conv_angle_unique)) / (
                    np.amax(conv_angle_unique) - np.amin(conv_angle_unique))).astype(np.float32)
        return conv_angle_norm

    # Scaling the PACBED by CNN
    def scale_pacbed(self, pacbed_measured, scale_const=None):
        # Create two images (smaller dimension for CNN input, larger dimension for Scaling)
        img_CNN, img_scaling = self.rescale_resize(pacbed_measured[:, :, np.newaxis], 1, self.dim)

        idx_pacbed, idx_conv = self.input_tensor_idx(self.scale_input_details)

        if scale_const is None:
            # Iterative scaling of the image by CNN
            k = 0
            scale_total = 1
            scale_pred = 1
            while True:
                # Transform image to RGB for CNN input
                img_CNN = np.tile(img_CNN, (1, 1, 3))

                # Input PACBED and normalized convergence angle in the correct format
                self.interpreter_scale.set_tensor(
                    self.scale_input_details[idx_conv]['index'],
                    self.conv_angle_norm[np.newaxis][np.newaxis, :]
                )
                self.interpreter_scale.set_tensor(
                    self.scale_input_details[idx_pacbed]['index'], img_CNN[np.newaxis, :, :, :]
                )

                # Interfere and make prediction
                self.interpreter_scale.invoke()
                scale_prediction = self.interpreter_scale.get_tensor(
                    self.scale_output_details[0]['index']
                )

                # Get scaling value with the highest predicted value
                new_scale_pred = self.label_scale['Scale / []'][np.argmax(scale_prediction)]

                # Damp prediction to avoid oscillating
                scale_pred = (k * scale_pred + (5 - k) * new_scale_pred) / 5

                # Break loop conditions if maximum iteration is reached or
                # prediction output is too small
                if k > 5 or np.amax(scale_prediction) < 0.5:
                    break
                    # raise RuntimeError("Could not predict scale")
                else:
                    img_CNN, img_scaling = self.rescale_resize(
                        img_scaling, scale_pred, self.dim[0:2]
                    )
                    # Loop counter
                    k += 1
                    scale_total *= scale_pred
        else:
            # Constant scaling with given value
            img_scaling, img_CNN = self.rescale_resize(img_scaling, scale_const, self.dim[0:2])
            img_CNN = np.tile(img_CNN, (1, 1, 3))

        return scale_total, img_CNN

    def rescale_resize(self, img, scale, dim):
        # Scale image (with full pixels)
        img = tf.keras.preprocessing.image.apply_affine_transform(
            img,
            zx=1 / scale,
            zy=1 / scale,
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode='constant',
            cval=-1.,
            order=1
        )

        # Resize and normalize image for next predictions
        img_cnn = tf.keras.preprocessing.image.smart_resize(img, dim[0:2], interpolation='bilinear')
        img_cnn = (
            2 * (img_cnn - np.amin(img_cnn)) / (np.amax(img_cnn) - np.amin(img_cnn)) - 1
        ).astype(np.float32)

        return (img_cnn, img)

    def predict(self, pacbed_measured: np.ndarray):

        assert len(pacbed_measured.shape) == 2

        # Preprocess PACBED
        # Subtract background
        pacbed_processed = background_subtraction(pacbed_measured)
        # Center PACBED
        pacbed_processed = center_PACBED(pacbed_processed)
        # Redim PACBED
        pacbed_processed = redim_PACBED(pacbed_processed, dim=(680, 680))

        # Scale PACBED
        scale_total, PACBED_scaled = self.scale_pacbed(pacbed_processed)

        # Make thickness prediction

        # Set input for CNN
        idx_pacbed, idx_conv = self.input_tensor_idx(self.thickness_input_details)
        self.interpreter_thickness.set_tensor(self.scale_input_details[idx_conv]['index'],
                                              self.conv_angle_norm[np.newaxis][np.newaxis, :])
        self.interpreter_thickness.set_tensor(self.scale_input_details[idx_pacbed]['index'],
                                              PACBED_scaled[np.newaxis, :, :, :])

        # Interfere and make prediction
        self.interpreter_thickness.invoke()
        thickness_cnn_output = self.interpreter_thickness.get_tensor(
            self.thickness_output_details[0]['index']
        )

        # Get thickness value with the highest predicted value
        thickness_predicted = self.label_thickness['Thickness / A'][
            np.argmax(thickness_cnn_output)
        ]

        # Make mistilt prediction

        # Set input for CNN
        idx_pacbed, idx_conv = self.input_tensor_idx(self.tilt_input_details)
        self.interpreter_tilt.set_tensor(
            self.scale_input_details[idx_conv]['index'],
            self.conv_angle_norm[np.newaxis][np.newaxis, :]
        )
        self.interpreter_tilt.set_tensor(
            self.scale_input_details[idx_pacbed]['index'], PACBED_scaled[np.newaxis, :, :, :]
        )

        # Interfere and make prediction
        self.interpreter_tilt.invoke()
        mistilt_cnn_output = self.interpreter_tilt.get_tensor(
            self.tilt_output_details[0]['index']
        )

        # Get mistilt value with the highest predicted value
        mistilt_predicted = self.label_mistilt['Mistilt / mrad'][np.argmax(mistilt_cnn_output)]

        result = {}
        result['thickness_pred'] = thickness_predicted
        result['thickness_cnn_output'] = thickness_cnn_output
        result['mistilt_pred'] = mistilt_predicted
        result['mistilt_cnn_output'] = mistilt_cnn_output
        result['scale'] = scale_total
        return result

    def validate(self, result, PACBED_measured, azimuth_i=0):

        # Create figure with special subplots
        fig, axs = plt.subplots(
            ncols=2, nrows=3, figsize=(8, 10), gridspec_kw={'height_ratios': [1.5, 1, 1]}
        )
        # Modifying subplots

        # First row
        gs = axs[-1, 0].get_gridspec()
        # Remove the underlying axes
        for ax in axs[-1, :]:
            ax.remove()
        ax1 = fig.add_subplot(gs[-1, :])

        # Second row
        gs = axs[-2, 0].get_gridspec()
        # Remove the underlying axes
        for ax in axs[-2, :]:
            ax.remove()
        ax2 = fig.add_subplot(gs[-2, :])

        # Sort thickness values (if labels are not ascended ordered)
        thickness_sort_ind = np.argsort(self.label_thickness.iloc[:, 0])
        thickness_pred_sorted = np.array(result['thickness_cnn_output'])[0, thickness_sort_ind]
        thickness_values_sorted = np.array(self.label_thickness.iloc[:, 0][thickness_sort_ind])

        # Sort mistilt values (if labels are not ascended ordered)
        mistilt_sort_ind = np.argsort(self.label_mistilt.iloc[:, 0])
        mistilt_pred_sorted = np.array(result['mistilt_cnn_output'])[0, mistilt_sort_ind]
        mistilt_values_sorted = np.array(self.label_mistilt.iloc[:, 0][mistilt_sort_ind])

        # Plot output of thickness prediction
        lineplot_thick = ax2.plot(
            thickness_values_sorted / 10, thickness_pred_sorted, linestyle='-',
            color='b', zorder=0
        )
        scatter_2 = ax2.plot(
            thickness_values_sorted[np.argmax(thickness_pred_sorted)] / 10,
            thickness_pred_sorted[np.argmax(thickness_pred_sorted)], marker='o', color='r'
        )
        ax2.set_title('Thickness Prediction')
        ax2.set_xlim([
            np.amin(thickness_values_sorted) / 10,
            np.amax(thickness_values_sorted) / 10
        ])
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Thickness / nm')

        # Plot output of mistilt prediction
        lineplot_tilt = ax1.plot(
            mistilt_values_sorted, mistilt_pred_sorted, linestyle='-', color='b', zorder=0
        )
        scatter_1 = ax1.plot(
            mistilt_values_sorted[np.argmax(mistilt_pred_sorted)],
            mistilt_pred_sorted[np.argmax(mistilt_pred_sorted)], marker='o', color='r'
        )
        ax1.set_title('Mistilt Prediction')
        ax1.set_xlim([np.amin(mistilt_values_sorted), np.amax(mistilt_values_sorted)])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Mistilt / mrad')

        # Plot loaded PACBED
        axs[0, 0].imshow(PACBED_measured)
        axs[0, 0].set_title('Measured PACBED')
        axs[0, 0].axis('off')

        # Plot best matching simulated PACBED

        # Get convergenc angle
        conv_angle_unique = np.unique(self.dataframe['Conv_Angle'])
        # Take the nearest simulated convergence angle
        conv_angle_plot = self.find_nearest(conv_angle_unique, self.conv_angle)

        # Get mistilt
        mistilt_i = np.argmax(mistilt_pred_sorted)
        mistilt = mistilt_values_sorted[mistilt_i]

        # Get thickness
        thickness_i = np.argmax(thickness_pred_sorted)
        thickness = thickness_values_sorted[thickness_i]

        path_img = self.create_path(self.dataframe, conv_angle_plot, thickness, mistilt, azimuth_i)
        PACBED_sim = self.load_img(path_img)

        # Plot simulated PACBED
        PACBED_sim_plot = axs[0, 1].imshow(PACBED_sim)
        axs[0, 1].set_title('Simulated PACBED')
        axs[0, 1].axis('off')

        # Add text
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        textstr = (
            r'Thickness = %.1f nm  ' % (thickness / 10,) +
            r'Mistilt = %.0f mrad  ' % (mistilt,) +
            r'Conv = %.0f mrad' % (conv_angle_plot,)
        )

        text = fig.text(
            0.5, 0.63, textstr, fontsize=14, horizontalalignment='center',
            verticalalignment='top', bbox=props
        )

        fig.tight_layout()

        f = io.BytesIO()
        plt.savefig(f, format='png')
        plt.close(fig)
        return f

    def filter_df(self, df, thickness, mistilt, conv):
        # Filter dataframe for the predicted values (open value is azimuth)
        filteredDataframe = df[
            (df['Thickness'] == thickness) &
            (df['Mistilt'] == mistilt) &
            (df['Conv_Angle'] == conv)
            ]
        # filteredDataframe.sort_values(by=['Azimuth'])

        filteredDataframe_sorted = filteredDataframe.sort_values(['Azimuth'], ascending=[True])

        return filteredDataframe_sorted.reset_index(drop=True)

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        if idx.size > 1:
            return array[idx][0]
        else:
            return array[idx]

    def create_path(self, df, conv_plot, thickness, mistilt, azimuth_i=0):

        filteredDataframe = self.filter_df(df, thickness, mistilt, conv_plot)

        path = filteredDataframe.iloc[azimuth_i]['Path'].replace('\\', '/')

        return path

    def load_img(self, path):
        # Load image
        img_sim = Image.open(path)

        img_sim_arr = np.array(img_sim)
        if len(img_sim_arr.shape) > 2:
            img_sim_arr = img_sim_arr[:, :, 0]

        return img_sim_arr
