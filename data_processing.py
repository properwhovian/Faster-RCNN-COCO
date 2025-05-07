# data_processing/data_preprocessing.py

import os
from pycocotools.coco import COCO
from PIL import Image

def load_coco_annotations(data_dir):
    """
    Load COCO dataset annotations.
    """
    annotations_file = os.path.join(data_dir, 'annotations_trainval2017', 'annotations', 'instances_train2017.json')
    coco = COCO(annotations_file)
    return coco

def get_category_names(coco):
    """
    Get all category names from the COCO dataset.
    """
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]
    return category_names

def get_image_info(coco, img_id):
    """
    Retrieve image information by ID.
    """
    img_info = coco.loadImgs(img_id)[0]
    return img_info

def load_image(img_path):
    """
    Load an image using PIL.
    """
    if os.path.exists(img_path):
        img = Image.open(img_path)
        return img
    else:
        print(f"Image file not found: {img_path}")
        return None
