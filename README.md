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

### Model Performance and Analyzing results on Skewed Dataset

Various Parameters for evaluation per 100 epochs are as follows in terms of PC,PSNR and SSIM on train and validation data:


| Epoch     |NIR_PC_train   |NIR_PC_val   |Red_PC_train   |Red_PC_val     |Green_PC_train  |Green_PC_val   |PSNR_train     |PSNR_val     |SSIM_train      |SSIM_val    |
| ----------|:-------------:| -----------:| ------------- |:-------------:| --------------:| ------------- |:-------------:| -----------:| -------------- |:-----------|
| 100       |-0.05000842315 |-0.0883398788| -0.014429011  |-0.01942083916 | 0.155553528    | 0.1314745827  |10.5787193     |9.944368826  | 0.1558209028   |0.1387626019|
| 200       |-0.09461359599 |-0.1647732402| 0.09893491829 |0.1078286654   | 0.07439906456  | 0.06439988178 |19.49975022    |17.96775821  | 0.5434919468   |0.4879971859|
| 300       |0.4844487891   |0.04423792917| -0.202653834  |0.2110942252   | 0.100536630    | 0.1016198901  |27.39418744    |24.93884501  | 0.9066141119   |0.8419657794|
| 400       |0.5659931852   |0.1553301583 | 0.3359637375  |0.2660727986   | 0.2462063361   | 0.1102983506  |31.2101439     |27.2062056   | 0.9494872523   |0.8889470705|
| 500       |0.1064666356   |0.0688902821 | 0.00386225793 |0.000663448184 | -0.06814932682 | -0.1012978036 |20.8824697     |19.21863675  | 0.4620478366   |0.4057640266|
| 600       |0.242182867    |0.1469791003 | 0.1299660457  |0.1374295475   |-0.02266885263  |-0.05183500483 |24.81609684    |22.2485777   |0.6618107623    |0.6026207847|
| 700       |0.1141179129   |0.09077203119|0.0231577498   |0.0461014954   |0.01413759484   |0.01533657492  |15.81737335    |14.91596596  |0.1561151655    |0.1371105281|
| 800       |0.07355812207  |0.07218999338|0.03420086299  |0.05645767604  |0.04626819331   |0.05379545084  |14.91536278    |14.151824    |0.1232767038    |0.1109237418|
| 900       |0.1402173311   |0.08612228292|0.08809589865  |0.0939511939   |0.1013317205    |0.09384005917  |18.03352817    |16.77826748  |0.2319571403    |0.2019726696|
| 1000      |0.1522175639   |0.03738729948|0.1245654264   |0.1658647433   |0.1536526258    |0.1253755692   |22.9342766     |20.83937953  |0.489397219     |0.4237837526|
| 1100      |0.2207068608   |0.08419658372|0.1730546559   |0.2173727323   |0.219995931     |0.1654665957   |26.96671849    |24.0358739   |0.7026804423    |0.6182513077|
| 1200      |0.2352971555   |0.0835442674 |0.1249354459   |0.147639293    |0.1379616916    |0.1153903569   |26.10884684    |23.56929742  |0.6732339429    |0.5950390088|
| 1300      |0.1791357889   |0.08168209727|0.008672194105 |0.03997902222  |0.0101701286    |0.006031576638 |23.91331973    |22.03335297  |0.5194873822    |0.4578776459|
| 1400      |0.3293007461   |0.1246761137 |0.02887394793  |0.05673247339  |0.04255831788   |0.01574076197  |23.80483363    |21.73766691  |0.5543500455    |0.4823385855|
| 1500      |0.3038193774   |0.06004512987|-0.001345626278|0.03361086924  |0.00718772725   |0.004563553405 |23.92568466    |21.85922651  |0.6341453641    |0.5565858715|
| 1600      |0.5292296154   |0.1225696892 |0.03089674103  |0.07625537081  |0.08322336163   |0.06492719505  |28.1318743     |24.72636908  |0.7972798953    |0.7104326953|
| 1700      |0.632629049    |0.1635108239 |0.08676428243  |0.1230198421   |0.2582400218    |0.1075018241   |31.25227794    |26.33953686  |0.8842468747    |0.7974064016|

#### Spikes in Skew Dataset Losses

The spikes in the generator loss graph can be due to a few reasons such as the image dataset might have a few big contrasts due to augmenting the dataset and the Generator might have been stuck in a mode collapse(local minima) which when resolved resulted in very high spikes in the loss as the descriminator was quite good trained till then.

## Contribution

[![Gitter](https://badges.gitter.im/Cloud-Removal-in-Satellite-Images-using-GANs/community.svg)](https://gitter.im/Cloud-Removal-in-Satellite-Images-using-GANs/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Please feel free to raise issues and fix any existing ones. Further details can be found in our [code of conduct](https://github.com/Chintan2108/Cloud-Removal-in-Satellite-Images-using-Conditional-Generative-Adversarial-Networks/blob/master/CODE_OF_CONDUCT.md).
