# INPUT

# Path and filename of dataframe (trained model and labels will be saved there)
path_df = 'D:/Github/Trained_Models/'
df_name = 'df.csv'

# Which cases should be trained (Thickness, Mistilt, Scale) - Train all cases in the vector
cases = ['Thickness', 'Mistilt', 'Scale']

# Dimension of the image for the CNN (for pretrained Xception-model --> (299,299,3) required) --> smaller grayscale images may be beneficial
dim = (299, 299, 3)

# Batch size
batch_size = 8

# Epochs for each case
epochs = {
  'Thickness': 30,
  'Mistilt': 25,
  'Scale': 5
}


# Optional validation of the data generator output and the trained model (time consuming - epoch runtime)
validation_datagenerator = 1
validation_model = 1



# Import required packages
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import os
import shutil

# Code Snippet limits GPU memory growth -> without, errors occur (may not necessary for cluster/other computers)
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)



# Load dataframe (csv-file with out index)
dataframe = pd.read_csv(path_df + df_name)

# Make custom data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, labels, case, dim=(299, 299, 1), batch_size=32, shuffle=True, scale_vec=None):
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.labels = labels
        self.num_classes = len(self.labels)
        self.shuffle = shuffle
        self.case = case
        self.dim = dim
        self.on_epoch_end()
        self.scale_vec = scale_vec

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y
    
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_label(self, label_id):
        # One Hot Encoding (maybe integer encoding better suitable)
        label_id = tf.keras.utils.to_categorical(label_id, self.num_classes)
        return label_id

    def __get_data(self, batch):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.num_classes))

        for i, id in enumerate(batch):

            # Loading image
            img = Image.open(self.df['Path'][id])
            img_arr = np.array(img)
            
            # If grayscale, add a dimension (required for tensorflow)
            if len(img.size) == 2:
                img_arr = np.array(img)[:, :, np.newaxis]
                
            # Crop image (removing empty space, depending on the simulation how much can be removed) --> remove already at saving the images (decreasing memory storage and loading time)
            img_arr =tf.image.central_crop(img_arr, central_fraction = 0.5)

            # Resize image to the required dimension
            img_arr = tf.keras.preprocessing.image.smart_resize(img_arr, self.dim[0:2], interpolation='bilinear')
            
            # Normalize image
            img_arr = img_arr / np.amax(img_arr)

            # Add noise: The larger the value the smaller the noise becomes, Poisson noise no negative values valid --> relu function
            img_arr = np.random.poisson(tf.nn.relu(img_arr) * 100) / 100

            # Loading Label for different cases (for scaling the label depends on the applied scale_rnd value)
            if self.case == 'Thickness':
                y_val = self.df['Thickness'][id]
            elif self.case == 'Mistilt':
                y_val = self.df['Mistilt'][id]
            elif self.case == 'Scale':
                # Make a random scaling operation
                scale_rnd = self.scale_vec[np.random.randint(len(self.scale_vec), size=1)[0]]
                # Scale image
                img_arr = tf.keras.preprocessing.image.apply_affine_transform(img_arr, zx=scale_rnd, zy=scale_rnd, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0., order=1)

                y_val = scale_rnd

            # Get categorical labels, one-hot encoded
            y[i,] = self.__get_label(np.array(self.labels.loc[self.labels.iloc[:, 0] == y_val, 'Index']))
            

            # Data Augmentation
            # in data generator --> CPU (Preprocessing in model as layers --> GPU, depending on the running system CPU or GPU calculations would be faster)
            
            # Random scaling only if scale is not trained (equal zooming in x and y, no straining)
            if self.case != 'Scale':
                zoom = np.random.uniform(0.9,1.1)
                img_arr = tf.keras.preprocessing.image.apply_affine_transform(img_arr, zx=zoom, zy=zoom, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0., order=1)

                
            # Random shear
            img_arr = tf.keras.preprocessing.image.random_shear(img_arr, intensity=0.05, row_axis=0, col_axis=1,
                                                                channel_axis=2, fill_mode='nearest', cval=0.0,
                                                                interpolation_order=1)
            
            # Random rotation (may change rotation from 45° to 90°)
            img_arr = tf.keras.preprocessing.image.random_rotation(img_arr, rg=45, row_axis=0, col_axis=1,
                                                                   channel_axis=2, fill_mode='nearest', cval=0.0,
                                                                   interpolation_order=1)
            
            # Random vertical and horizontal shift
            img_arr = tf.keras.preprocessing.image.random_shift(img_arr, wrg=0.1, hrg=0.1, row_axis=0, col_axis=1,
                                                                channel_axis=2, fill_mode='nearest', cval=0.0,
                                                                interpolation_order=1)
            # Random flip left/right and up/down
            if random.choice([0, 1]):
                img_arr = tf.image.flip_left_right(img_arr)
            if random.choice([0, 1]):
                img_arr = tf.image.flip_up_down(img_arr)

            # Filling batch
            X[i,] = img_arr

        return X, y

# Training of all elements of cases
for k in range(0,len(cases)):

    # Extract training case
    case = cases[k]


    # Preparing Dataframe for training and validation
    df_train, df_validation = train_test_split(dataframe, test_size=0.2)

    # Determine labels for data generator in a dataframe-format
    if case == 'Thickness':
        label_unique = np.unique(dataframe['Thickness'])
        df_labels = pd.DataFrame({'Thickness / A' : label_unique,'Index' : np.arange(0,len(label_unique))})
        scale_vec = None
    elif case == 'Mistilt':
        label_unique = np.unique(dataframe['Mistilt'])
        df_labels = pd.DataFrame({'Mistilt / mrad' : label_unique,'Index' : np.arange(0,len(label_unique))})
        scale_vec = None
    elif case == 'Scale':
        scale_vec = [0.5, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2, 2.5]
        label_unique = np.unique(scale_vec)
        df_labels = pd.DataFrame({'Scale / []' : label_unique,'Index' : np.arange(0,len(label_unique))})


    # Create DataGenerator for training and validation
    datagenerator_train = DataGenerator(df_train, df_labels, case, dim, batch_size, shuffle=True, scale_vec=scale_vec)
    datagenerator_validation = DataGenerator(df_validation, df_labels, case, dim, batch_size, shuffle=False, scale_vec=scale_vec)


    # Create Model

    # Fully connected layers with dropout
    fc_layers = [1024, 1024]
    dropout = 0.3

    # Load pretrainerd xception model (can be changed with other models or non-pretrained model)
    base_model = tf.keras.applications.Xception(weights='imagenet',
                                                include_top=False,
                                                input_shape=dim)
    # Build model
    inputs = tf.keras.Input(shape=dim)
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    for fc in fc_layers:
        x = tf.keras.layers.Dense(fc, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(datagenerator_train.num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # Callback functions for early stopping and saving the best model should be implemented

    # Compile  model
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()]) # other optimizers for faster convergence could be tested

    # Print summary of the model
    model.summary()


    # Train model
    history_train = model.fit(datagenerator_train,
                              epochs=epochs[case],
                              validation_data=datagenerator_validation
                              )



    # OUTPUT

    path_save = path_df

    # Save model as tensorflow framework (where dataframe is located) -- introduce temporay solution
    tf.saved_model.save(model, os.path.join(path_save, 'Xception_' + case))
    # Convert to tensorflow lite framework
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path_save, 'Xception_' + case)) # path to the SavedModel directory
    tflite_model = converter.convert()
    
    # Save the model.
    with open(os.path.join(path_save, 'Xception_' + case + '.tflite'), 'wb') as f:
        f.write(tflite_model)
      
    # Delete tensorflow saved model
    shutil.rmtree(os.path.join(path_save, 'Xception_' + case))
      

    # Save labels (where dataframe is located)
    df_labels.to_csv(os.path.join(path_save, "Xception_" + case + '_labels.csv'),index=False)





    # VALIDATION

    # Plot data generator output with the corresponding labels (check data augmentation)
    if validation_datagenerator == 1:

        import matplotlib.pyplot as plt

        # Number of plots
        rows = 3
        cols = 3

        # Loading images with datagenerator
        datagenerator_plot = DataGenerator(dataframe, df_labels, case, dim, batch_size=rows*cols, shuffle=True, scale_vec=scale_vec)
        datagen_plot = iter(datagenerator_plot)
        imgs = next(datagen_plot)

        # Plotting
        axes=[]
        fig=plt.figure()

        for j in range(rows*cols):
            # Add subplot
            axes.append(fig.add_subplot(rows, cols, j+1) )

            # Add tilte (label)
            if case == 'Thickness':
                appendix = ' / A'
            elif case == 'Mistilt':
                appendix = ' / mrad'
            elif case == 'Scale':
                appendix = ''

            subplot_title=(case + ' ' + str(df_labels.iloc[np.argmax(imgs[1][j]), 0]) + appendix)
            axes[-1].set_title(subplot_title)

            # Plot image
            plt.imshow(imgs[0][j,:,:,:])

            axes[-1].axis('off')
        fig.tight_layout()
        plt.show()


    # Plot of the confusion matrix (see if predicted values are far from true values, or close to true value)
    if validation_model == 1:

        from sklearn.metrics import confusion_matrix
        import seaborn as sn
        import matplotlib.pyplot as plt

        # Data generator for loading images
        datagenerator_confusion = DataGenerator(dataframe, df_labels, case, dim, batch_size, shuffle=False, scale_vec=scale_vec)

        # Predict images and save predicted and true values
        datagenerator_matrix = iter(datagenerator_confusion)
        y_pred = []
        y_true = []
        for j in range(0,len(dataframe)//batch_size):
            img_pred, y_true_gen = next(datagenerator_matrix)
            Y_pred = model.predict(img_pred)
            y_pred = np.append(y_pred, np.argmax(Y_pred, axis=1))
            y_true = np.append(y_true, np.argmax(y_true_gen, axis=1))

        # Generate confusion matrix
        confusion_m = confusion_matrix(y_true, y_pred,labels=df_labels['Index'])

        # Plot in seaborn
        df_cm = pd.DataFrame(confusion_m, index = [i for i in df_labels.iloc[:, 0]],
                          columns = [i for i in df_labels.iloc[:, 0]])
        # qualtitative check if predicted values are far from diagonal (set annot to true for more information)
        plt.figure()
        sn.heatmap(df_cm, annot=False)
