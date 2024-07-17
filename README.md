# Classification of 101 food classes using ResNet, YOLO and Falcon 2 #

WORK IN PROGRESS

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

consts.py -> has all the constants <br>
utils.py -> helper functions for the project <br>
requirements.txt -> pip-based dependencies <br>
Data -> contains the dataset <br>
Food-Vision.ipynb -> main experimentation Jupyter notebook <br>

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

----------------------------
CLASSES: 

 ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'] 

----------------------------
DATASET SIZE: 4.77 GiB images 
