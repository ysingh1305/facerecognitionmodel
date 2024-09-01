import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import keras
import cv2
import os 
import random 
from keras import models, Model
from keras import Layer, layers
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')





anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(100)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(100)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(100)

def preprocess(filepath):
    byte_img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img/255.0
    return img

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


def make_embedding():
    inp = keras.Input(shape=(100,100,3), name='input_image')
    c1 = layers.Conv2D(64, (10,10), activation = 'relu')(inp)
    m1 = layers.MaxPooling2D(64, (2,2), padding = 'same')(c1)
    c2 = layers.Conv2D(128, (7,7), activation = 'relu')(m1)
    m2 = layers.MaxPooling2D(64, (2,2), padding = 'same')(c2)
    c3 = layers.Conv2D(128, (4,4), activation = 'relu')(m2)
    m3 = layers.MaxPooling2D(64, (2,2), padding = 'same')(c3)
    c4 = layers.Conv2D(256, (4,4), activation = 'relu')(m3)
    f1 = layers.Flatten()(c4)
    d1 = layers.Dense(4096, activation = 'sigmoid')(f1)


    return keras.Model(inputs= [inp], outputs= [d1], name='embedding')
embedding = make_embedding()
print(embedding.summary())


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        # Ensure that input_embedding and validation_embedding are tensors, not lists
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        
        # Compute the L1 distance
        return tf.math.abs(input_embedding - validation_embedding)



def make_siamese_model(): 
    # Anchor image input in the network
    input_image = keras.Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = keras.Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer.name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = layers.Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
print(siamese_model.summary())

binary_cross_loss = keras.losses.BinaryCrossentropy()

opt = keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        
        # Reshape yhat to match the shape of y
        yhat = tf.squeeze(yhat, axis=-1)
        yhat = tf.squeeze(yhat, axis=0)
        
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # Return loss
    return loss

def train(data, EPOCHS):
    for epoch in range(1, EPOCHS+1):
        progbar = keras.utils.Progbar(len(train_data))
        
        for idx, batch in enumerate(train_data):
            train_step(batch)
            progbar.update(idx+1)
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50

train(train_data, EPOCHS=EPOCHS)

from keras import metrics

test_input, test_val, y_true = test_data.as_numpy_iterator().next()

predictions = siamese_model.predict([test_input, test_val])


m = metrics.Recall()
m.update_state(y_true, predictions)
print(m.result())

m = metrics.Precision()

# Calculating the recall value 
m.update_state(y_true, predictions)

# Return Recall Result
print(m.result())


siamese_model.save('siamesemodel.h5')
