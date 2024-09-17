import os
import json
import xmltodict
from tqdm import tqdm
import pandas as pd

def order_dict(dictionary):
    result = {}
    for k, v in sorted(dictionary.items()):
        if isinstance(v, dict):
            result[k] = order_dict(v)
        else:
            result[k] = v
    return result

def process_object(object):
    df = pd.json_normalize([object], sep='_')
    # flatten objects
    object = order_dict(df.to_dict(orient='records')[0])
    # fix types
    object['bndbox_xmax'] = round(float(object['bndbox_xmax']))
    object['bndbox_ymax'] = round(float(object['bndbox_ymax']))
    object['bndbox_xmin'] = round(float(object['bndbox_xmin']))
    object['bndbox_ymin'] = round(float(object['bndbox_ymin']))

    return object

def convert_xml_to_json_lines(input_folder, output_file):
    with open(output_file, 'w') as jsonl_file:
        for filename in tqdm(os.listdir(input_folder)):
            if filename.endswith('.xml'):
                xml_file_path = os.path.join(input_folder, filename)
                
                with open(xml_file_path, 'r') as xml_file:
                    try:
                        # Convert XML to dictionary
                        xml_dict = order_dict(xmltodict.parse(xml_file.read())['annotation'])
                        # force objects to be a list
                        xml_dict['object'] = xml_dict['object'] if type(xml_dict['object']) == list else [xml_dict['object']]
                        # flatten objects
                        df = pd.json_normalize([xml_dict], sep='_')
                        xml_dict = df.to_dict(orient='records')[0]
                        # fix types
                        xml_dict['segmented'] = True if xml_dict['segmented'] == '1' else False
                        xml_dict['size_depth'] = int(xml_dict['size_depth'])
                        xml_dict['size_height'] = int(xml_dict['size_height'])
                        xml_dict['size_width'] = int(xml_dict['size_width'])
                        # fix objects
                        xml_dict['object'] = [process_object(object) for object in xml_dict['object']]

                        # Convert dictionary to JSON string, and write to output file
                        json_line = json.dumps(xml_dict)
                        jsonl_file.write(json_line + '\n')
                        
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")


if __name__ == "__main__":
    input_folder = 'data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations'  # Replace with the path to your XML folder
    output_file = 'data/dataset.jsonl'              # Output JSONL file
    convert_xml_to_json_lines(input_folder, output_file)
    print(f"Conversion completed. JSON Lines saved to {output_file}.")
