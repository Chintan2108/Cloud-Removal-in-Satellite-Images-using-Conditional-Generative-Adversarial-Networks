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

### Model and Training 

pix-2-pix GAN model is used for learning the mapping of the cloudy image to it's cloud-free counterpart. It follows a supervised conditional vector based training approach. The model is trained on the cloudy and cloud-free image pairs. pix-2-pix model follows a pixel to pixel image restoration approach and generates a new pixel for every cloudy pixel.

### Architectures 

#### Generator Architecture (U-Net)

![Proposed U-Net Generator for the pix2pix GAN](/network architectures/generator_new__.png)

#### Discriminator Architecture  (PatchGAN)

![Proposed PatchGAN Discriminator for the pix2pix GAN](/network architectures/discriminator_.png)

## Contribution

Please feel free to raise issues and fix any existing ones. Further details can be found in our [code of conduct](https://github.com/Chintan2108/Cloud-Removal-in-Satellite-Images-using-Conditional-Generative-Adversarial-Networks/blob/master/CODE_OF_CONDUCT.md).
