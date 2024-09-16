import os
import json
import xmltodict
from tqdm import tqdm

def order_dict(dictionary):
    result = {}
    for k, v in sorted(dictionary.items()):
        if isinstance(v, dict):
            result[k] = order_dict(v)
        else:
            result[k] = v
    return result

def convert_xml_to_json_lines(input_folder, output_file):
    with open(output_file, 'w') as jsonl_file:
        for filename in tqdm(os.listdir(input_folder)):
            if filename.endswith('.xml'):
                xml_file_path = os.path.join(input_folder, filename)
                
                with open(xml_file_path, 'r') as xml_file:
                    try:
                        # Convert XML to dictionary
                        xml_dict = order_dict(xmltodict.parse(xml_file.read())['annotation'])
                        
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
