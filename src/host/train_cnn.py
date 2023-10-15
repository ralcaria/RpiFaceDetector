import os
from collections import namedtuple
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

@dataclass
class Dataset:
    imagedata: np.ndarray
    labels: np.ndarray

def read_dataset(path, img_height, img_width) -> Dataset:
    """Lookup the image paths and extract the actual images

    Args:
        path (str): Path to the dataset csv file
        img_height (int): Height of image
        img_width (int): Width of image

    Returns:
        Dataset: Object containing the images and target labels
    """
    TARGETS = {'NotReigan': 0, 'Reigan': 1}
    imagedata = []
    labels = []

    with open(path, 'r') as f:
        cols = f.readline().rstrip()
        ImageGroup = namedtuple('ImageGroup', cols)
        for line in f.readlines():
            imagepath,label = line.rstrip().split(',')
            img_grp = ImageGroup(imagepath, label)

            images = tf.keras.preprocessing.image.load_img(img_grp.imagepath, target_size=(img_height, img_width), color_mode='grayscale') #convert RGB to grayscale image
            images = tf.keras.preprocessing.image.img_to_array(images) / 255 #scale pixel values between 0 and 1

            imagedata.append(images)
            labels.append(TARGETS[img_grp.label])

        
    return Dataset(
        imagedata=np.array(imagedata, dtype=np.float32),
        labels=np.array(labels),
    )

def main():
    """Get the images --> Build the model --> Train the model --> Test the model --> Save the model

    """  
    INPUTDIR = os.path.join(os.getcwd(), '..', '..', 'input')
    HEIGHT = 320
    WIDTH = 480

    #Read the train and validation dataset
    train_dataset = read_dataset(os.path.join(INPUTDIR, 'training_dataset.csv'), HEIGHT, WIDTH)
    test_dataset = read_dataset(os.path.join(INPUTDIR, 'validation_dataset.csv'), HEIGHT, WIDTH)

    #Create the CNN architecture
    input_layer = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, 1))
    conv_layer = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(input_layer)
    conv_layer = tf.keras.layers.MaxPooling2D()(conv_layer)
    conv_layer = tf.keras.layers.Conv2D(32, (2, 2), activation='relu')(conv_layer)
    conv_layer = tf.keras.layers.Flatten()(conv_layer)

    #Add object classification layer
    clf_layer = tf.keras.layers.Dense(32, activation='relu')(conv_layer)
    clf_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='clf_out')(clf_layer) #binary classification

    #Fit the train data to the CNN
    model = tf.keras.models.Model(inputs=input_layer, outputs=clf_layer)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics='accuracy')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    print(model.summary())

    model.fit(train_dataset.imagedata,
            train_dataset.labels,
            validation_data=(test_dataset.imagedata, test_dataset.labels),
            callbacks=callback,
            batch_size=5,
            epochs=20,
            shuffle=True,
            verbose=1)

    #Validate the model accuracy
    raw_preds = model.predict(test_dataset.imagedata)
    preds = np.where(raw_preds >= 0.5, 1, 0).ravel()
    tn, fp, fn, tp = confusion_matrix(test_dataset.labels, preds).ravel()
    print(tn, fp, fn, tp)

    #Save the model to disk
    output_path = os.path.join("..", "output", "face_detect_model.h5")
    model.save(output_path, save_format="h5")
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert() #reduce the size of the model by converting it to tflite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    return None

if __name__ == '__main__':
    main()
