from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from keras.layers import AveragePooling2D, Concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras import Model, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
import cv2
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.callbacks import ModelCheckpoint

'''
Load Data
'''

def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory 
    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory
    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids       = labels_pd.id.values
    data      = []
    for identifier in ids:
        fname     = dir_data + identifier.astype(str) + '.jpg'
        image     = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data) # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids
    pass


dir_train_images  = './training_resized/'
dir_test_images   = './validation_resized/'
dir_train_labels  = './labels_training.csv'
dir_test_labels   = './labels_validaton.csv'

X_train, y_train = load_data(dir_train_images, dir_train_labels, training=True)

X_train_rgb = []
for img in X_train:
    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    X_train_rgb.append(img_rgb)
X_train_rgb = np.array(X_train_rgb)

X_train_rgb = (X_train_rgb)/255

y_train_cat = to_categorical(y_train, num_classes = 4)

'''
split the training dataset into train/validation set
'''

from sklearn.model_selection import train_test_split
X_training, X_test, y_training, y_test = train_test_split(X_train_rgb, y_train_cat, test_size=0.3, random_state=1428)

X_training = X_training.reshape(-1,224,224,3)
X_test = X_test.reshape(-1,224,224,3)

'''
Pretrained Model
'''

### VGG ###

# vgg_pretrained_model = VGG16(weights="imagenet", 
#                              include_top= False,
#                              input_tensor=Input(shape=(224, 224, 3)))
# new_model = vgg_pretrained_model.output

### ResNet50 ###

resnet50_pretrained = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

new_model = resnet50_pretrained.output

new_model = AveragePooling2D(pool_size=(4, 4))(new_model)
new_model = Flatten(name="flatten")(new_model)
new_model = Dense(64, activation="relu")(new_model)
new_model = Dropout(0.4)(new_model)
new_model = Dense(4, activation="softmax")(new_model)    

# model = Model(inputs=vgg_pretrained_model.input, outputs=new_model)
model = Model(inputs=resnet50_pretrained.input, outputs=new_model)


'''
Data Augmentation
'''

from keras.callbacks import EarlyStopping

BS = 8
EPOCHS = 50

opt = Adam(lr=0.0002, decay=0.003 / (EPOCHS))
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

class_weight = {0: 3.,
                1: 4.,
                2: 7.,
                3: 30.}

aug = ImageDataGenerator(rotation_range = 2)

# translation, rotation, horizontal flip, and intensity shift

'''
Model fit
'''
np.random.seed(1028)
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
H = model.fit_generator(aug.flow(X_training, y_training, batch_size=BS),
                        validation_data=(X_test, y_test), steps_per_epoch=len(X_train) // BS, epochs=EPOCHS, verbose=1,
                        class_weight = class_weight, callbacks = [mcp_save])


'''
prediction
'''
score = model.predict(np.array(X_train_rgb))

y_preds = np.argmax(score, axis=1)

accuracy_score(y_train.ravel(), y_preds)
confusion_matrix(y_train.ravel(), y_preds)

pd.crosstab(y_train, y_preds, rownames=['True'], colnames=['Predicted'], margins=True)


'''
save model
'''
filename = 'cnn_r50_aug_lr2e4_cwa3a4a7a30.sav'
pickle.dump(model, open(filename, 'wb'))

'''
load model
'''
filename = 'cnn_r50_aug_lr2e4_cwa3a4a7a30.sav'
loaded_model = pickle.load(open(filename, 'rb'))


'''
Test on the validation_resized folder, but I may saved wrong file format so I need to fix the data
'''
X_val, y_val = load_data(dir_test_images, dir_test_labels, training=True)

X_val_rgb = []
for img in X_val:
    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    X_val_rgb.append(img_rgb)
X_val_rgb = np.array(X_val_rgb)

X_val_rgb = (X_val_rgb)/255

y_val_cat = to_categorical(y_val, num_classes = 4)