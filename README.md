# Classification of 101 food classes using ResNet, YOLO and Falcon 2

Main idea and focus was to compare CNN architectures such as ResNet and YOLO vs a vision-language model (VLM).

Dataset: food101 

This dataset consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.
