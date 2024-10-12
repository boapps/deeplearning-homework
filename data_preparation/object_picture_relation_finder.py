import os
import xml.etree.ElementTree as ET
import json
from collections import Counter
from tqdm import tqdm

def create_json_file(object_names_count, folder, filename):
    json_filename = "_".join([f"{obj}{count}" for obj, count in object_names_count.items()]) + ".json"
    json_filepath = os.path.join(folder, json_filename)

    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {"fileNames": []}
    
    if filename not in data["fileNames"]:
        data["fileNames"].append(filename)

    with open(json_filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def process_xml_file(xml_path, output_folder):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text

    object_names = [obj.find('name').text for obj in root.findall('object')]

    object_names_count = Counter(object_names)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    create_json_file(object_names_count, output_folder, filename)

def iterate_xml_folder(input_folder, output_folder):
    # Get the list of XML files in the folder
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    
    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(input_folder, xml_file)
        process_xml_file(xml_path, output_folder)

if __name__ == "__main__":
    input_folder = '../data/VOCdevkit/VOC2012/Annotations'
    output_folder = '../data/relations'
    if os.path.exists(output_folder):
        print('relations exists, exiting')
    else:
        iterate_xml_folder(input_folder, output_folder)
