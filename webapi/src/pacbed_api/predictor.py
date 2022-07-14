import glob
import os
import io
import threading

# from ncempy.io import dm
import numpy as np
from skimage import filters
from skimage.measure import regionprops
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar

from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

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


def background_subtraction(pacbed_img, sigma_fac):
    # Subtract Background (improves prediction at larger thicknesses)
    background = gaussian_filter(pacbed_img, sigma=sigma_fac*pacbed_img.shape[0], mode = 'constant', cval = 0)

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
    PACBED_img = PACBED_img.resize(dim, resample=Image.BILINEAR)

    print(f'PACBED dimensions changed to {dim}')

    return np.asarray(PACBED_img)


class Predictor:
    def __init__(self, parameters_prediction, num_threads):
        # Declare variables
        self.id_system = parameters_prediction['id_system']
        self.id_model = parameters_prediction['id_model']

        self.dataframe = None
        self.dim = None

        self.interpreters = threading.local()
        self.num_threads = num_threads

        # Scale CNN
        self._scale_path = None
        # Thickness CNN
        self._thickness_path = None
        # Mistilt CNN
        self._tilt_path = None
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

    def _get_or_create_interp(self, name, path):
        if not hasattr(self.interpreters, name):
            interp = tf.lite.Interpreter(
                model_path=path, num_threads=self.num_threads,
            )
            interp.allocate_tensors()
            setattr(self.interpreters, name, interp)
        return getattr(self.interpreters, name)

    @property
    def interpreter_scale(self) -> tf.lite.Interpreter:
        return self._get_or_create_interp("scale", self._scale_path)

    @property
    def scale_input_details(self):
        return self.interpreter_scale.get_input_details()

    @property
    def scale_output_details(self):
        return self.interpreter_scale.get_output_details()

    @property
    def interpreter_thickness(self) -> tf.lite.Interpreter:
        return self._get_or_create_interp("thickness", self._thickness_path)

    @property
    def thickness_input_details(self):
        return self.interpreter_thickness.get_input_details()

    @property
    def thickness_output_details(self):
        return self.interpreter_thickness.get_output_details()

    @property
    def interpreter_tilt(self) -> tf.lite.Interpreter:
        return self._get_or_create_interp("tilt", self._tilt_path)

    @property
    def tilt_input_details(self):
        return self.interpreter_tilt.get_input_details()

    @property
    def tilt_output_details(self):
        return self.interpreter_tilt.get_output_details()

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
                self._scale_path = os.path.join(self.Path_models, model_name)
            elif model_name.find('Thickness') > -1:
                self._thickness_path = os.path.join(self.Path_models, model_name)
            elif model_name.find('Mistilt') > -1:
                self._tilt_path = os.path.join(self.Path_models, model_name)

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

        print('Models and Labels loaded.')

    def input_tensor_idx(self, input_details):
        if len(input_details[0]['shape']) == 4:
            idx_pacbed = 0
            idx_conv = 1
        else:
            idx_pacbed = 1
            idx_conv = 0

        return idx_pacbed, idx_conv

    def get_conv_angle_norm(self, conv_angle):
        # Used convergence angle
        conv_angle_unique = np.unique(self.dataframe['Conv_Angle'])
        # Calculate scaled convergence angle
        conv_angle_norm = ((conv_angle - np.amin(conv_angle_unique)) / (
                    np.amax(conv_angle_unique) - np.amin(conv_angle_unique))).astype(np.float32)
        return conv_angle_norm

    # Scaling the PACBED by CNN
    def scale_pacbed(self, pacbed_measured, conv_angle_norm, scale_const=None):
        # Speed up scaling algorithm (working with CNN input dimension)
        pacbed_scaling = redim_PACBED(pacbed_measured, dim=self.dim[0:2])[:, :, np.newaxis]

        # Zoom image (first time to get the correct format)
        img_CNN = self.rescale_resize(pacbed_scaling, 1, self.dim)

        idx_pacbed, idx_conv = self.input_tensor_idx(self.scale_input_details)

        if scale_const is None:
            # Iterative scaling of the image by CNN to get a final total scale value
            k = 0
            scale_total = 1

            while True:
                # Transform image to RGB for CNN input
                img_CNN = np.tile(img_CNN, (1, 1, 3))

                # Input PACBED and normalized convergence angle in the correct format
                self.interpreter_scale.set_tensor(
                    self.scale_input_details[idx_conv]['index'],
                    conv_angle_norm[np.newaxis][np.newaxis, :]
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
                scale_pred = (k * 1 + (5 - k) * new_scale_pred) / 5

                # Break loop conditions if maximum iteration is reached or
                # prediction output is too small
                if k > 5 or np.amax(scale_prediction) < 0.8 or new_scale_pred == 1:
                    break
                    # raise RuntimeError("Could not predict scale")
                else:
                    # Scale image with the total scale factor
                    scale_total *= scale_pred
                    img_CNN = self.rescale_resize(
                        pacbed_scaling, scale_total, self.dim[0:2])
                    # Loop counter
                    k += 1
        else:
            # Constant scaling with given value
            scale_total = scale_const

        return scale_total

    def rescale_resize(self, img, scale, dim):
        # Scale image (with full pixels)
        img = tf.keras.preprocessing.image.apply_affine_transform(
            img,
            zx=1 / scale,
            zy=1 / scale,
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode='nearest',
            #cval=-1.,
            order=1
        )

        # Resize and normalize image for next predictions
        img_cnn = tf.keras.preprocessing.image.smart_resize(img, dim[0:2], interpolation='bilinear')
        img_cnn = (
            2 * (img_cnn - np.amin(img_cnn)) / (np.amax(img_cnn) - np.amin(img_cnn)) - 1
        ).astype(np.float32)

        return img_cnn

    def predict(self, pacbed_measured: np.ndarray, conv_angle: float):

        assert len(pacbed_measured.shape) == 2

        # Preprocess PACBED
        pacbed_non_zero = pacbed_measured.copy()
        pacbed_non_zero[pacbed_non_zero < 0] = 0

        # Redim PACBED (downsizing to speed up operation, upsizing to match CNN input dimension)
        if pacbed_non_zero.size > 680 * 680:
            # Downscaling
            pacbed_processed = redim_PACBED(pacbed_non_zero, dim=(680, 680))
            # Blur PACBED
            pacbed_processed = gaussian_filter(pacbed_processed, sigma=3) # Prediction is sensitive to this
        elif pacbed_non_zero.size < 200 * 200:
            # Upscaling
            pacbed_processed = redim_PACBED(pacbed_non_zero, dim=self.dim[0:2])
        else:
            # no scaling
            pacbed_processed =pacbed_non_zero


        # Center PACBED
        pacbed_processed = center_PACBED(pacbed_processed)

        # Subtract background
        pacbed_processed = background_subtraction(pacbed_processed, sigma_fac = 0.9)
        
        # Normalize convergenc angle
        conv_angle_norm = self.get_conv_angle_norm(conv_angle)

        # Scale PACBED
        scale_total = self.scale_pacbed(pacbed_processed, conv_angle_norm, scale_const = None)
        PACBED_scaled = self.rescale_resize(pacbed_processed[:, :, np.newaxis], scale_total, self.dim[0:2])
        PACBED_scaled = np.tile(PACBED_scaled, (1, 1, 3))

        # Make thickness prediction

        # Set input for CNN
        idx_pacbed, idx_conv = self.input_tensor_idx(self.thickness_input_details)
        self.interpreter_thickness.set_tensor(self.scale_input_details[idx_conv]['index'],
                                              conv_angle_norm[np.newaxis][np.newaxis, :])
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
            conv_angle_norm[np.newaxis][np.newaxis, :]
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
        return result, PACBED_scaled[:,:,0]

    def validate(self, result, pacbed_pred_out, conv_angle, azimuth_i=0):

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

        # Plot best matching simulated PACBED

        # Get convergenc angle
        conv_angle_unique = np.unique(self.dataframe['Conv_Angle'])
        # Take the nearest simulated convergence angle
        conv_angle_plot = self.find_nearest(conv_angle_unique, conv_angle)

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
        

        # Rotate measured PACBED to match simulated PACBED
        pacbed_pred_out = (pacbed_pred_out - np.amin(pacbed_pred_out))/(np.amax(pacbed_pred_out) - np.amin(pacbed_pred_out))
        rot, scale, pacbed_measured = self.polar_registration(pacbed_pred_out, PACBED_sim)
        PACBED_measured = np.asarray(Image.fromarray(pacbed_measured).rotate(-rot, fillcolor = int(np.amin(pacbed_measured))))
        PACBED_measured = center_PACBED(PACBED_measured)
        
        axs[0, 0].imshow(PACBED_measured)
        axs[0, 0].set_title('Measured PACBED')
        axs[0, 0].axis('off')
        

        # Add text
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        textstr = (
            r'Thickness = %.1f nm  ' % (thickness / 10,) +
            r'Mistilt = %.0f mrad  ' % (mistilt,) +
            r'Conv = %.1f mrad' % (conv_angle_plot,)
        )

        text = fig.text(
            0.5, 0.63, textstr, fontsize=14, horizontalalignment='center',
            verticalalignment='top', bbox=props
        )

        fig.tight_layout()

        f = io.BytesIO()
        fig.savefig(f, format='png')
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

    def polar_registration(self, pacbed_measured, pacbed_sim):

        # Finding mistilt direction by identifiying quadrant with the highest intensity
        h = len(pacbed_measured)
        w = len(pacbed_measured[1])
        top_left = np.sum([pacbed_measured[i][:w // 2] for i in range(h // 2)])
        top_right = np.sum([pacbed_measured[i][w // 2:] for i in range(h // 2)])
        bot_left = np.sum([pacbed_measured[i][:w // 2] for i in range(h // 2, h)])
        bot_right = np.sum([pacbed_measured[i][w // 2:] for i in range(h // 2, h)])

        quadrants = [top_left, top_right, bot_left, bot_right]
        if np.argmax(quadrants) == 0:
            k = 0
        elif np.argmax(quadrants) == 1:
            k = 1
        elif np.argmax(quadrants) == 2:
            k = 2
        elif np.argmax(quadrants) == 3:
            k = 3
        # Rotate PACBED, so that mistilt shows in upper left corner
        pacbed_measured = np.rot90(pacbed_measured, k=k, axes=(0, 1))



        radius = pacbed_sim.shape[0]
        image_polar = warp_polar(pacbed_sim, radius=radius,
                                 scaling='log')
        rescaled_polar = warp_polar(pacbed_measured, radius=radius,
                                    scaling='log')

        # setting `upsample_factor` can increase precision
        shifts, error, phasediff = phase_cross_correlation(image_polar, rescaled_polar,
                                                           upsample_factor=2)
        shiftr, shiftc = shifts[:2]

        klog = radius / np.log(radius)
        shift_scale = 1 / (np.exp(shiftc / klog))

        return shiftr, shift_scale, pacbed_measured
