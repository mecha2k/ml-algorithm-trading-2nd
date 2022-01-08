#!/usr/bin/env python
# coding: utf-8

# # How to extract bottleneck features

# Modern CNNs can take weeks to train on multiple GPUs on ImageNet, but fortunately, many researchers share their final weights. Keras, e.g., contains pre-trained models for several of the reference architectures discussed above, namely VGG16 and 19, ResNet50, InceptionV3 and InceptionResNetV2, MobileNet, DenseNet, NASNet and MobileNetV2
# 
# This notebook illustrates how to download pre-trained VGG16 model, either with the final layers to generate predictions or without the final layers as illustrated in the figure below to extract the outputs produced by the bottleneck features.

# ## Imports

# In[1]:


import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K


# In[2]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')


# ## Load and Preprocess Sample Images
# 
# Before supplying an image to a pre-trained network in Keras, there are some required preprocessing steps.
# 
# We have imported a very small dataset of 7
# images and stored the  preprocessed image input as `img_input`.  Note that the dimensionality of this array is `(8, 224, 224, 3)`.  In this case, each of the 8 images is a 3D tensor, with shape `(224, 224, 3)`.

# In[3]:


img_paths = Path('images/img_input').glob('*.jpg')


# In[4]:


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# In[5]:


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


# In[6]:


# calculate the image input
img_input = preprocess_input(paths_to_tensor(img_paths))

img_input.shape


# ## Import Pre-Trained VGG-16
# 
# Import the VGG-16 network (including the final classification layer) that has been pre-trained on ImageNet.

# Keras makes it very straightforward to download and use pre-trained models:

# In[7]:


vgg16 = VGG16()
vgg16.summary()


# For this network, `model.predict` returns a 1000-dimensional probability vector containing the predicted probability that an image returns each of the 1000 ImageNet categories.  The dimensionality of the obtained output from passing `img_input` through the model is `(8, 1000)`.  The first value of `7` merely denotes that 7 images were passed through the network.

# In[8]:


y_pred = vgg16.predict(img_input)
y_pred.shape


# In[9]:


np.argmax(y_pred, axis=1)


# ## Import the VGG-16 Model, with the Final Fully-Connected Layers Removed
# 
# When performing transfer learning, we need to remove the final layers of the network, as they are too specific to the ImageNet database.  This is accomplished in the code cell below.
# 
# ![VGG-16 model for transfer learning](images/vgg19_transfer.png)

# You can use this model like any other Keras model for predictions. To exclude the fully-connected layers, just add the keyword `include_top=False` to obtain the output of the final convolutional layer when passing an image to the CNN.

# In[10]:


vgg16 = VGG16(include_top=False)
vgg16.summary()


# By omitting the fully-connected layers, we are no longer forced to use a fixed input size for the model (224x224, the original ImageNet format). By only keeping the convolutional modules, our model can be adapted to arbitrary input sizes.

# ### Extract Output of Final Max Pooling Layer
# 
# Now, the network stored in `model` is a truncated version of the VGG-16 network, where the final three fully-connected layers have been removed.  In this case, `model.predict` returns a 3D array (with dimensions $7\times 7\times 512$) corresponding to the final max pooling layer of VGG-16.  The dimensionality of the obtained output from passing `img_input` through the model is `(8, 7, 7, 512)`.  The first value of `8` merely denotes that 8 images were passed through the network.  

# In[11]:


vgg16.predict(img_input).shape


# This is exactly how we calculate the bottleneck features for your project!

# ## Import ResNet50

# ### With final layer

# In[12]:


resnet = ResNet50()
resnet.summary()


# ### Without final layer

# In[13]:


resnet = ResNet50(include_top=False)
resnet.summary()


# ## Import Inception V3

# ### With final layer

# In[14]:


inception = InceptionV3()
inception.summary()


# ### Without final layer

# In[15]:


inception = InceptionV3(include_top=False)
inception.summary()

