#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary functions 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#loading test data
# src_data = np.load(ROOT_PATH + '/data/test_data_3_10copy.npy')
src_data = np.load("C:\\Users\\verma\\Downloads\\4 angles orientation\\4 angles orientation\\test_data_4_10copy.npy")

#loading train data
# tar_data = np.load(ROOT_PATH + '/data/train_data_3_10copy.npy')
tar_data = np.load("C:\\Users\\verma\\Downloads\\4 angles orientation\\4 angles orientation\\train_data_4_10copy.npy")

print("Train data shape: ", src_data.shape)
print("Test data shape: ", tar_data.shape)


# ## Train Data

# In[3]:


tar_data[1].shape


# In[4]:


img1=tar_data[1]
img2=tar_data[2]
img3=tar_data[3]
img4=tar_data[4]


# In[5]:


img5=src_data[1]
img6=src_data[2]
img7=src_data[3]
img8=src_data[4]


# In[6]:


# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 
datagen = ImageDataGenerator(  
        shear_range = 0.3, 
        zoom_range = 0.3,) 


# In[7]:


x1 = img1
# Reshaping the input image 
x1 = x1.reshape((1, ) + x1.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x1, batch_size = 1): 
    i += 1
    if i >1: 
        break      
j1=datagen.flow(x1,batch_size=1)
    
x2 = img2
# Reshaping the input image 
x2 = x2.reshape((1, ) + x2.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x2, batch_size = 1): 
    i += 1
    if i >1: 
        break
j2=datagen.flow(x2,batch_size=1)   

x3 = img3
# Reshaping the input image 
x3 = x3.reshape((1, ) + x3.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x3, batch_size = 1): 
    i += 1
    if i >1: 
        break
j3=datagen.flow(x3,batch_size=1)
        
x4 = img4
# Reshaping the input image 
x4 = x4.reshape((1, ) + x4.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x4, batch_size = 1): 
    i += 1
    if i >1: 
        break
j4=datagen.flow(x4,batch_size=1)


# In[8]:


j1=j1.next()
j2=j2.next()
j3=j3.next()
j4=j4.next()


# In[9]:


j1.shape=(1024,1024,3)
j2.shape=(1024,1024,3)
j3.shape=(1024,1024,3)
j4.shape=(1024,1024,3)


# In[10]:


fig, a = plt.subplots(2,4, figsize=(16,8))

a[0][0].imshow(j1)
a[0][0].set_title('original skew')
a[0][0].axis('off')

a[0][1].imshow(j2)
a[0][1].set_title('-90-deg skew')
a[0][1].axis('off')

a[0][2].imshow(j3)
a[0][2].set_title('180-deg skew')
a[0][2].axis('off')

a[0][3].imshow(j4)
a[0][3].set_title('+90-deg skew')
a[0][3].axis('off')

a[1][0].imshow(img1)
a[1][0].set_title('original')
a[1][0].axis('off')

a[1][1].imshow(img2)
a[1][1].set_title('-90-deg')
a[1][1].axis('off')

a[1][2].imshow(img3)
a[1][2].set_title('180-deg')
a[1][2].axis('off')

a[1][3].imshow(img4)
a[1][3].set_title('+90-deg')
a[1][3].axis('off')

plt.show()


# In[11]:


#stacking the augmented True images
train_stack = np.stack((img1, j1, img2, j2, img3, j3, img4, j4, img1, j1), axis=0)
print(train_stack.shape)


# ## TEST DATA

# In[12]:


x5 = img5
# Reshaping the input image 
x5 = x5.reshape((1, ) + x5.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x5, batch_size = 1): 
    i += 1
    if i >1: 
        break      
j5=datagen.flow(x5,batch_size=1)
    
x6 = img6
# Reshaping the input image 
x6 = x6.reshape((1, ) + x6.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x6, batch_size = 1): 
    i += 1
    if i >1: 
        break
j6=datagen.flow(x6,batch_size=1)   

x7 = img7
# Reshaping the input image 
x7 = x7.reshape((1, ) + x7.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x7, batch_size = 1): 
    i += 1
    if i >1: 
        break
j7=datagen.flow(x7,batch_size=1)
        
x8 = img8
# Reshaping the input image 
x8 = x8.reshape((1, ) + x8.shape)  
# Generating augmented samples using the above defined parameters.  
i = 0
for batch in datagen.flow(x8, batch_size = 1): 
    i += 1
    if i >1: 
        break
j8=datagen.flow(x8,batch_size=1)


# In[13]:


j5=j5.next()
j6=j6.next()
j7=j7.next()
j8=j8.next()

j5.shape=(1024,1024,3)
j6.shape=(1024,1024,3)
j7.shape=(1024,1024,3)
j8.shape=(1024,1024,3)


# In[14]:


fig, a = plt.subplots(2,4, figsize=(16,8))

a[0][0].imshow(j5)
a[0][0].set_title('original skew')
a[0][0].axis('off')

a[0][1].imshow(j6)
a[0][1].set_title('-90-deg skew')
a[0][1].axis('off')

a[0][2].imshow(j7)
a[0][2].set_title('180-deg skew')
a[0][2].axis('off')

a[0][3].imshow(j8)
a[0][3].set_title('+90-deg skew')
a[0][3].axis('off')

a[1][0].imshow(img5)
a[1][0].set_title('original')
a[1][0].axis('off')

a[1][1].imshow(img6)
a[1][1].set_title('-90-deg')
a[1][1].axis('off')

a[1][2].imshow(img7)
a[1][2].set_title('180-deg')
a[1][2].axis('off')

a[1][3].imshow(img8)
a[1][3].set_title('+90-deg')
a[1][3].axis('off')

plt.show()


# In[15]:


#stacking the augmented FALSE images
test_stack = np.stack((img5, j5, img6, j6, img7, j7, img8, j8, img5, j5), axis=0)
print(test_stack.shape)


# In[16]:


#saving the training set on disk
np.save('C:\\Users\\verma\\Downloads\\4 angles orientation\\4 angles orientation\\train_data_copy', train_stack)
#saving the testing set on disk
np.save('C:\\Users\\verma\\Downloads\\4 angles orientation\\4 angles orientation\\test_data_copy', test_stack)


# In[ ]:




