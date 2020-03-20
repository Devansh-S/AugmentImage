
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import math


# In[ ]:


class Augment_Image():
    def __init__(self, path):
        self.path = path
    '''self.height = height
        self.width = width 
        self.method = method
        self.padding = padding'''


    def read_image(directory):
        self.image_directory = directory
        for filename in tqdm(os.listdir(self.image_directory)):
            path = os.path.join(self.image_directory, filename)
            image = plt.imread(path)
            return image

        
    def pad_image(image, x_size, y_size, color=[0.0, 0.0, 0.0, 1.0]):
        s = img.shape
        x_size = x_size - s[0]
        if x_size%2 == 0:
            TOP_SIZE = int(x_size/2)
            BOTTOM_SIZE = int(x_size/2)
        else:
            TOP_SIZE = int(math.ceil(x_size/2))
            BOTTOM_SIZE = int(math.floor(x_size/2))

        y_size = y_size - s[1]
        if y_size%2 == 0:
            LEFT_SIZE = int(y_size/2)
            RIGHT_SIZE = int(y_size/2)
        else:
            LEFT_SIZE = int(math.ceil(y_size/2))
            RIGHT_SIZE = int(math.floor(y_size/2))

        if x_size > 0 or y_size > 0:
            image = cv2.copyMakeBorder(img, TOP_SIZE, BOTTOM_SIZE, LEFT_SIZE, RIGHT_SIZE, cv2.BORDER_CONSTANT, value=color) 
        return image

    
    def stretch_image(img, x_size, y_size):
        shape = img.shape
        x = shape[1]
        y = shape[0]
        x = x_size/x
        y = y_size/y
        affine_warp = np.array([[x, 0, 0], [0, y, 0]], dtype=np.float32)
        dsize = (x_size, y_size)
        warped_img = cv2.warpAffine(img, affine_warp, dsize)
        return warped_img
        
    def random_cropping():
        
    
    
    
    def random_rotation():
        random_degree = random.uniform(-25, 25)
        return sk.transform.rotate(image_array, random_degree)
        
        
        
    def random_flipping():
        
        
        
        
    def random_brightness():
        
        
        
        
    def random_zoom():
    
      
        
        
    def hv_shift():
        
        
        
        
    def h_flip(image):
        return image[:, ::-1]
        
    def v_flip(image):
        return image[:, ::-1]

    def random_noise(image_array: ndarray):
        # add random noise to the image
        return sk.util.random_noise(image_array)

    


# In[142]:


def read_image(directory):
    image_directory = directory
    for filename in tqdm(os.listdir(image_directory)):
        path = os.path.join(image_directory, filename)
        image = plt.imread(path)
        return image

def pad_image(image, x_size, y_size, color=[0.0, 0.0, 0.0, 1.0]):
    
    s = img.shape
    
    x_size = x_size - s[0]
    if x_size%2 == 0:
        TOP_SIZE = int(x_size/2)
        BOTTOM_SIZE = int(x_size/2)
    else:
        TOP_SIZE = int(math.ceil(x_size/2))
        BOTTOM_SIZE = int(math.floor(x_size/2))
    
    y_size = y_size - s[1]
    if y_size%2 == 0:
        LEFT_SIZE = int(y_size/2)
        RIGHT_SIZE = int(y_size/2)
    else:
        LEFT_SIZE = int(math.ceil(y_size/2))
        RIGHT_SIZE = int(math.floor(y_size/2))
        
    if x_size > 0 or y_size > 0:
        image = cv2.copyMakeBorder(img, TOP_SIZE, BOTTOM_SIZE, LEFT_SIZE, RIGHT_SIZE, cv2.BORDER_CONSTANT, value=color) 
    return image

def stretch_image(img, x_size, y_size):
    shape = img.shape
    x = shape[1]
    y = shape[0]
    x = x_size/x
    y = y_size/y
    affine_warp = np.array([[x, 0, 0], [0, y, 0]], dtype=np.float32)
    dsize = (x_size, y_size)
    warped_img = cv2.warpAffine(img, affine_warp, dsize)
    return warped_img


# In[253]:


img = plt.imread('testing.png')


# In[258]:


img.shape[2]


# In[250]:


img = stretch_image(img,300,300)[:,:,0:3]
img2 = np.array(img)
img2.shape
plt.imshow(img)


# In[172]:


p = np.load('Downloads/data.npy',)


# In[217]:


img1 = img


# In[218]:


img1 = img[:,:,0:3]


# In[219]:


img1 = [img1]


# In[220]:


img1 = np.array(img1)


# In[221]:


img1.shape


# In[251]:


img = plt.imread('Downloads/download.jpeg')


# In[252]:


plt.imshow(img1)
img.shape


# In[92]:


img = np.array(img)
img = img.resize((100,100))
plt.imshow(img)


# In[72]:


img = pad_image(img, 300, 300)
plt.imsave('new.jpg',img)
plt.imshow(img)
img.shape


# In[68]:


print(img[140][140])


# In[150]:


# example of brighting image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


# In[222]:


img = load_img('testing.png')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
#samples = expand_dims(data, 0)
#plt.imshow(samples)
# create image data augmentation generator
# prepare iterator
it = datagen.flow(img1, batch_size=1)
# generate samples and plot
plt.figure(figsize=(14,14))
for i in range(36):
    # define subplot
    pyplot.subplot(6, 6, 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    print(np.mean(image))
# show the figure
pyplot.show()


# In[223]:


datagen = ImageDataGenerator(featurewise_center=False,
                                             samplewise_center=False,
                                             featurewise_std_normalization=False,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,
                                             zca_epsilon=1e-06,
                                             rotation_range=60,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             brightness_range=[0.5,1.5],
                                             shear_range=0.3,
                                             zoom_range=0.0,
                                             channel_shift_range=0.0,
                                             fill_mode='nearest',
                                             cval=0.0,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             rescale=None,
                                             preprocessing_function=None,
                                             data_format='channels_last',
                                             validation_split=0.0,
                                             dtype='float32')


# In[231]:


save_stretched = save_directory + batch + 'aug-stretch-'+str(i)


# In[232]:


print(save_stretched)


# In[225]:


save_directory = 'train/'


# In[227]:


image = 'image000'
label = 'bird'
it = 1


# In[230]:


batch = image+'.'+label+'.'+str(it)
print(batch)


# In[233]:


save_stretched - '-35'


# In[242]:


a = 'a.c.v.d.v'


# In[243]:


import os


# In[247]:


print(os.path.splitext(a)[1])

