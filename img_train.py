# Required imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


class Parameters:
    def __init__(self):
        self.TRAIN_DIR = r"C:\Users\Harish\Desktop\human activity recognition\train"
        self.IMG_HEIGHT = 112
        self.IMG_WIDTH = 112
        self.BATCH_SIZE = 32
        self.EPOCHS = 10
        self.NUM_CLASSES = 2
        self.LR = 0.001


param = Parameters()


def build_model(param):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(param.IMG_HEIGHT, param.IMG_WIDTH, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(param.NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=param.LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    param.TRAIN_DIR,
    target_size=(param.IMG_HEIGHT, param.IMG_WIDTH),
    batch_size=param.BATCH_SIZE,
    class_mode='categorical')
model = build_model(param)
print("[INFO] training the model...")
history = model.fit(
    train_generator,
    epochs=param.EPOCHS,
    steps_per_epoch=train_generator.samples // param.BATCH_SIZE)
model.save('activity_recognition_model.h5')
print("[INFO] model trained and saved as 'activity_recognition_model.h5'")
