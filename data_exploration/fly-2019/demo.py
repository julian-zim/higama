import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

'''
To open an image in .nii format
We use nibabel package to read image.
The details on how to install nibabel can be found here: http://nipy.org/nibabel/
'''
image_path = '' # The file path of the image of interest; the image should be in .nii format
img = nib.load(image_path)
img_data = img.get_data()


'''
Calculate the average of images and plot mean value over time.
'''
M = np.mean(np.mean(np.mean(img_data,0),0),0)
plt.plot(M)
plt.show()



'''
Plot the middle voxel of each image over time.
'''
voxel = img_data[img_data.shape[0]/2,img_data.shape[1]/2,img_data.shape[2]/2,:]
plt.plot(voxel)
plt.show()