# Classification of 101 food classes using ResNet, YOLO and Falcon 2 #

Main idea and focus was to compare CNN architectures such as ResNet and YOLO vs a vision-language model (VLM).

## Workflow: ##

1. A very basic CNN was trained to set a base line
2. ResNet with ImageNet weights was partially trained via transfer learning by training last 5 layers, while freezing all the layers above.
3. YOLOv8 model was trained on the dataset
4. Automatic pipeline was built for Falcon 2 to classify images via zero-shot prompting.

## Results: ##

| Model   | Accuracy | Loss | F1 |
|---------|----------|------|----|
| Basic   |          |      |    |
| ResNet  |          |      |    |
| YOLO    |          |      |    |
| Falcon2 |          |      |    |

## Files: ##

consts.py -> has all the constants
utils.py -> helper functions for the project
requirements.txt -> pip-based dependencies
Data -> contains the dataset
Food-Vision.ipynb -> main experimentation Jupyter notebook

## Usage: ##

In case you want to reproduce the results:

Requirements: Python 3.11+, Unix-based OS.

1. `git clone https://github.com/akvachan/Food-Vision.git`
2. `cd Food-Vision`
3. Check Python 3.11+: `python --version`
4. `pip install -r requirements.txt`
5. `jupyter-lab .`
6. Open "Food-Vision.ipynb" in Jupyter File Browser
7. Run the notebook

## Dataset: food101 ##

This dataset consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.
