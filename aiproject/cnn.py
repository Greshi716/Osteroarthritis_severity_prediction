from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier=Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(224,224,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=5,activation='softmax'))
classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('osteroarthritis/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('osteroarthritis/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 80,
                         epochs = 150,
                         validation_data = test_set,
                         validation_steps = 32)

classifier.save("modelf.h5")
print("Saved model to disk")