import os
import json
from tqdm import tqdm
import shutil

def copy_images_and_masks(relation_folder, image_dir, mask_dir, augmented_dir, target_size=(128, 128), num_classes=21):
    collected_image_names = set()

    json_files = [f for f in os.listdir(relation_folder) if f.endswith('.json')]
    for json_file in tqdm(json_files, desc="Collecting image names from JSON files"):
        json_path = os.path.join(relation_folder, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            file_names = data.get("fileNames", [])
            for file_name in file_names:
                base_name = os.path.splitext(file_name)[0]
                collected_image_names.add(base_name)
    
    for base_name in tqdm(sorted(collected_image_names), desc="Loading images and masks from relations"):
        img_path = os.path.join(image_dir, f"{base_name}.jpg")
        img_path_out = './img'
        mask_path = os.path.join(mask_dir, f"{base_name}.png")
        mask_path_out = './msk'
        
        if os.path.exists(img_path):
            shutil.copy(img_path, img_path_out)
        
        if os.path.exists(mask_path):
            shutil.copy(mask_path, mask_path_out)
            
    
    augmented_files = os.listdir(augmented_dir)
    augmented_base_names = {f.split('-')[0] for f in augmented_files}
    
    for base_name in tqdm(sorted(augmented_base_names), desc="Loading augmented images and masks"):
        img_files = [f for f in augmented_files if f.startswith(base_name)]
        for img_file in img_files:
            aug_img_path = os.path.join(augmented_dir, img_file)
            img_path_out = './img'
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            mask_path_out = './msk'
            
            if os.path.exists(aug_img_path):
                shutil.copy(aug_img_path, img_path_out)
            
            if os.path.exists(mask_path):
                shutil.copy(mask_path, mask_path_out)

relation_folder = '../data/relations'
image_dir = '../data/VOCdevkit/VOC2012/JPEGImages'
mask_dir = '../data/VOCdevkit/VOC2012/SegmentationClass'
augmented_dir = '../data/VOCdevkit/VOC2012/AugmentedImages'

os.mkdir('img')
os.mkdir('msk')

copy_images_and_masks(relation_folder, image_dir, mask_dir, augmented_dir)