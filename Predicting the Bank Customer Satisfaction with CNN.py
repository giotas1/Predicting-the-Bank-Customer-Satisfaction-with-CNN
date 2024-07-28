#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/competitions/santander-customer-satisfaction/data

# # Step 1: Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Step 2: Importing dataset 

# In[3]:


dataset= pd.read_csv('train.csv')


# In[4]:


dataset.head()


# # Step 3: Data Preprocessing

# In[ ]:


dataset.shape


# In[ ]:


# independent variables (Matrix of features)
x = dataset.drop(labels=['ID','TARGET'], axis=1)


# In[ ]:


# dependent variable
y = dataset['TARGET']


# In[ ]:


x.shape, y.shape


# In[ ]:


# splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[ ]:


x_train.shape, x_test.shape


# # Step 4: Remove constant, Quasi constant and duplicate features

# In[ ]:


from sklearn.feature_selection import VarianceThreshold


# In[ ]:


rm_f = VarianceThreshold(threshold=0.01)
x_train = rm_f.fit_transform(x_train)
x_test = rm_f.transform(x_test)


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


369-266


# In[ ]:


# remove duplicate features
x_train_t = x_train.T
x_test_t = x_test.T


# In[ ]:


x_train_t = pd.DataFrame(x_train_t)
x_test_t = pd.DataFrame(x_test_t)


# In[ ]:


x_train_t.shape, x_test_t.shape


# In[ ]:


x_train_t.duplicated()


# In[ ]:


# number of duplicate features
x_train_t.duplicated().sum()


# In[ ]:


duplicated_features = x_train_t.duplicated()
print(duplicated_features)


# In[ ]:


features_to_keep = [not index for index in duplicated_features]
print(features_to_keep)


# In[ ]:


x_train = x_train_t[features_to_keep].T
x_test = x_test_t[features_to_keep].T


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


266-250


# In[ ]:


# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


x_train


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


# reshape the dataset
x_train = x_train.reshape(60816, 250, 1)
x_test = x_test.reshape(15204, 250, 1)


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# # Step 5: Building the model

# In[ ]:


# define an object
model = tf.keras.models.Sequential()


# In[ ]:


# first CNN layer
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape = (250, 1)))

# batch normalization
model.add(tf.keras.layers.BatchNormalization())

# maxpool layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))

# dropout layer
model.add(tf.keras.layers.Dropout(0.3))


# In[ ]:


# second CNN layer
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))

# batch normalization
model.add(tf.keras.layers.BatchNormalization())

# maxpool layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))

# dropout layer
model.add(tf.keras.layers.Dropout(0.5))


# In[ ]:


# third CNN layer
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))

# batch normalization
model.add(tf.keras.layers.BatchNormalization())

# maxpool layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))

# dropout layer
model.add(tf.keras.layers.Dropout(0.5))


# In[ ]:


# flatten layer
model.add(tf.keras.layers.Flatten())


# In[ ]:


# first dense layer (fully connected layer)
model.add(tf.keras.layers.Dense(units=256, activation='relu'))

# dropout layer
model.add(tf.keras.layers.Dropout(0.5))


# In[ ]:


# output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


opt = tf.keras.optimizers.Adam(learning_rate=0.00005)


# In[ ]:


# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# # Step 6: Training the model

# In[ ]:


history =  model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# In[ ]:


# model predictions
y_pred = model.predict_classes(x_test)


# In[ ]:


print(y_pred[12]), print(y_test[12])


# In[ ]:


# confusion matrix
from sklearn.metrics import  confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


acc_cm = accuracy_score(y_test, y_pred)


# In[ ]:


print(acc_cm)


# # Step 7: Learning Curve

# In[ ]:


def learning_curve(history, epoch):

  # training vs validation accuracy
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'val'], loc='upper left')
  plt.show()

  # training vs validation loss
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'val'], loc='upper left')
  plt.show()


# In[ ]:


learning_curve(history, 10)

