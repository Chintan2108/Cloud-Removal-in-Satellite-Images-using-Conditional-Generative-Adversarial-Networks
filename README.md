# Cloud-Removal-in-Satellite-Images-using-Conditional-Generative-Adversarial-Networks

## Affiliation
#### Photogrammetry and Remote Sensing Department, Indian Institute of Remote Sensing, ISRO, Dehradun
#### April 2020 - July 2020

## Summary

Cloud cover in the earth's atmosphere is a major issue in temporal optical satellite image processing. Clouds, thick or thin, cover the earth features in a satellite image and hide important information. The objective of this project is to develop a deep learning based automated pipeline to remove cloud cover from optical satellite imagery. Conditional GANs architecture is used to learn the mapping of the cloudy satellite image to it's cloud-free counterpart. A novel augmented and computationally efficient training approach is suggested.

## Modules

### Built With/Libraries Used/Network Architectures

* Keras (TF Backend)
* Numpy
* Matplotlib
* PatchGAN
* U-Net

### Dataset

Sentinel-2 is a part of the opensource earth-observation *Copernicus* mission of the ESA. It provides high temporal resolution multispectral data (13 bands) at 10m spatial resolution. False Color Composite Images (Green, Red and NIR bands) are used. The study site is chosen based availability of phenological features (crops/agriculture). 
Removing cloud cover in Sentinel-2 satellite images using only optical data and a novel augmented training approach using conditional GANs.

Currently the training dataset is augmented from a single pair of cloudy and cloud-free images labelled as True and False image and used as Training and Test Data respectively.

The initial augmentation is performed only by rotation of 0, 90, 180 and -90 degrees which gives us 4 pairs, say 1,2,3 and 4.These four pairs are then stacked iteratively 10 times as: [1,2,3,4,1,2,3,4,1,2].
Later on, more data augmentation was implemented by performing skew operations on the cloudy and cloud-free images, and then perform the rotation operations. Now there are 4 pairs of original (unskewed) images, say, [1, 2, 3, 4] and 4 pairs of the skewed images, say [1', 2', 3', 4'] which are stacked as [1,2,3,4,1',2',3',4',1,1'].

[Training Data Drive Link](https://drive.google.com/file/d/1dJqHhZUHgYrf0_42CBswJklQgxsApAXQ/view?usp=sharing)

[Skew Augmented Training Data Drive Link](https://drive.google.com/file/d/1jipx2uAULbB32oSfW09k6rJmMdkoh_Wt/view?usp=sharing)

### Model and Training 

pix-2-pix GAN model is used for learning the mapping of the cloudy image to it's cloud-free counterpart. It follows a supervised conditional vector based training approach. The model is trained on the cloudy and cloud-free image pairs which are dated two days apart. pix-2-pix model follows a pixel to pixel image restoration approach and generates a new pixel cloud-free for every cloudy pixel.

[Trained Model Drive Link](https://drive.google.com/drive/folders/1kmXGRGnV4Lj8zRIoLWPp7FLGhNDf7bKQ?usp=sharing)

### Architectures 

#### Discriminator Architecture (PatchGAN)

![Proposed U-Net Generator for the pix2pix GAN](https://github.com/Chintan2108/Cloud-Removal-in-Satellite-Images-using-Conditional-Generative-Adversarial-Networks/blob/master/network%20architectures/discriminator_.png)

#### Generator Architecture (U-Net)

![Proposed PatchGAN Discriminator for the pix2pix GAN](https://github.com/Chintan2108/Cloud-Removal-in-Satellite-Images-using-Conditional-Generative-Adversarial-Networks/blob/master/network%20architectures/generator_new__.jpg)

## Contribution

[![Gitter](https://badges.gitter.im/Cloud-Removal-in-Satellite-Images-using-GANs/community.svg)](https://gitter.im/Cloud-Removal-in-Satellite-Images-using-GANs/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Please feel free to raise issues and fix any existing ones. Further details can be found in our [code of conduct](https://github.com/Chintan2108/Cloud-Removal-in-Satellite-Images-using-Conditional-Generative-Adversarial-Networks/blob/master/CODE_OF_CONDUCT.md).
