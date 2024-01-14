
import csv
import os
import re


def create_class_mapping(class_mapping_path):
    class_mapping = {}
    with open(class_mapping_path, 'r') as class_mapping_file:
        content = class_mapping_file.read()

        
        matches = re.findall(r'name:\s*"([^"]+)"\s*id:\s*(\d+)', content, re.DOTALL)


        for match in matches:
            class_name, class_id = match
            class_mapping[class_name.strip()] = int(class_id)
    return class_mapping
def convert_csv_to_yolo(csv_path, output_dir, class_mapping):
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            image_path = row['file']
            class_name = row['name']
            class_id = class_mapping.get(class_name)
            
            if class_id is None:
                print(f"Warning: Class '{class_name}' not found in class mapping. Skipping.")
                continue

            x_min, y_min, x_max, y_max = map(float, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            image_width = float(row['width'])
            image_height = float(row['height'])

            yolo_format = convert_to_yolo_format(class_id, x_min, y_min, x_max, y_max, image_width, image_height)
            output_file_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '.txt'))

            with open(output_file_path, 'a') as output_file:
                output_file.write(yolo_format)


def convert_to_yolo_format(class_id, x_min, y_min, x_max, y_max, image_width, image_height):
    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


csv_path = 'C:\\Users\\Hüseyin Erdem Erol\\Desktop\\1 DL Project\\traffic_signs_dataset_v1\\train.csv'
output_dir = 'C:\\Users\\Hüseyin Erdem Erol\\Desktop\\1 DL Project\\traffic_signs_dataset_v1\\output_yolo_files'
class_mapping_path = 'C:\\Users\\Hüseyin Erdem Erol\\Desktop\\1 DL Project\\traffic_signs_dataset_v1\\label_map.pbtxt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class_mapping = create_class_mapping(class_mapping_path)
convert_csv_to_yolo(csv_path, output_dir, class_mapping)
print("Conversion complete.")


