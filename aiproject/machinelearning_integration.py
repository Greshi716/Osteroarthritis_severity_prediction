import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
model = load_model('modelf.h5')
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('osteroarthritis/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory('osteroarthritis/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
train_data = []
train_labels = []
num_train_samples = len(train_generator.filenames)
num_test_samples=len(test_generator.filenames)
for i in range(num_train_samples // 32):
    batch = train_generator.next()
    features = model.predict(batch[0])
    train_data.append(features)
    train_labels.append(batch[1])

train_data = np.concatenate(train_data, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

test_data = []
test_labels = []

for i in range(num_test_samples // 32):
    batch = test_generator.next()
    features = model.predict(batch[0])
    test_data.append(features)
    test_labels.append(batch[1])

test_data = np.concatenate(test_data, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

rf = RandomForestClassifier(n_estimators=200, random_state=42)

rf.fit(train_data, np.argmax(train_labels, axis=1))

y_pred = rf.predict(test_data)

accuracy = accuracy_score(np.argmax(test_labels, axis=1), y_pred)
# print("Accuracy:", accuracy)
