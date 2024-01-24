# Instance segmentation of people

### Prequisites

In order to run the network you will have to download the pre-trained YOLO weight file (237 MB). https://pjreddie.com/media/files/yolov3.weights and put in in `/models/yolo` folder.

### Dataset

Dataset used for this project is based on [CityScapes](https://www.cityscapes-dataset.com/login/).

From CityScapes:

> We present a new large-scale dataset that contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames. The dataset is thus an order of magnitude larger than similar previous attempts. Details on annotated classes and examples of our annotations are available at this webpage.

Three packages from CityScapes were used thorought this project:

- gtFine_trainvaltest.zip (241MB) [md5]
- leftImg8bit_trainvaltest.zip (11GB) [md5]
- gtBbox_cityPersons_trainval.zip (2.2MB) [md5]

Dataset has been suited to segmentation of people and made smaller in `step0_preparation.ipynb`. Moreover, also in `step0_preparation.ipynb`, yolo detection have been used to transform the dataset unto patches as to follow the method from [this paper](https://openreview.net/forum?id=NZ4LUn1g9-).

`step1_UNet_training.ipynb` is where the VGG-UNet architecture was trained using yolo patches dataset. Experiments were run on Colab.

Finally, in `step2_instance_segmentation.ipynb`, using pretrained YOLO and trained VGG-UNet models, instance segmentation is performed.

### Citations

- M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, "The Cityscapes Dataset for Semantic Urban Scene Understanding," in CVPR 2016.
- S. Zhang, R. Benenson, and B. Schiele, "CityPersons: A Diverse Dataset for Pedestrian Detection," in CVPR 2017.
- Bizhe Bai, Jie Tian, Tao Wang, Sicong Luo, & Sisuo LYU. (2022). YUSEG: Yolo and Unet is all you need for cell instance segmentation.

Project by Weronika Zawadzka & Dominika Plewińska.
