from ultralytics import YOLO
import os
import cv2
import math

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Calculate intersection area
    intersection_area = max(0, min(x2, x2_gt) - max(x1, x1_gt)) * max(0, min(y2, y2_gt) - max(y1, y1_gt))

    # Calculate union area
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = area_box1 + area_box2 - intersection_area

    # Avoid division by zero
    epsilon = 1e-5

    # Calculate IoU
    iou = intersection_area / (union_area + epsilon)

    return iou

def count_false_negatives(labels_directory, image_file, result):
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_directory, label_file)

    # Check if the label file exists
    if os.path.exists(label_path):
        with open(label_path, 'r') as label_file:
            lines = label_file.readlines()
            ground_truth_boxes = [list(map(float, line.strip().split()[1:])) for line in lines]

            # Count false negatives
            false_negatives = len(ground_truth_boxes)

            for info in result:
                boxes = info.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    predicted_box = [x1, y1, x2, y2]

                    # Check if predicted box matches any ground truth box
                    for gt_box in ground_truth_boxes:
                        iou = calculate_iou(predicted_box, gt_box)
                        if iou > 0.3:
                            false_negatives -= 1
                            break

            return false_negatives
    else:
        return 0

# Load the YOLO model
model = YOLO('best.pt')

# Reading the classes
classnames = ['30km-h-Hiz-Limiti-Speed-limit-', '50km-h-Hiz-Limiti-Speed-limit-', '60km_h Hiz Limiti (Speed limit)',
              '70km-h-Hiz-Limiti-Speed-limit-', '90km-h-Hiz-Limiti-Speed-limit-',
              'Bir Sonraki Kavsakta Gecis Hakki (Right-of-way at the next intersection)', 'Buzlanma (Beware of ice/snow)',
              'Dur (Stop)', 'Duz veya Saga Gidis (Go straight or right)', 'Duz veya Sola Gidis (Go straight or left)',
              'Engebeli Yol (Bumpy road)', 'Gecis Yasagi Sonu (End of no passing)', 'Gecis Yasagi (No passing)',
              'Genel Uyari (General caution)', 'Saga Surekli Tehlikeli Viraj (Double curve-Right)',
              'Kaygan yol (Slippery road)', 'Oncelikli Yol (Priority road)',
              'Saga Tehlikeli Viraj (Dangerous curve to the right)', 'Sagdan Gidiniz (Keep right)',
              'Sola Tehlikeli Viraj (Dangerous curve to the left)', 'Soldan Gidiniz (Keep left)',
              'Trafik Isiklari (Traffic signals)', 'Yayalar (Pedestrians)', 'Yol Calismasi (Road work)', 'Yol Ver (Yield)',
              'Zorunlu Doner Kavsak (Roundabout mandatory)', 'U Donusu (U-turn)', 'U Donusu Yasak (U Turn is not allowed)',
              'Gevsek Yamac (Loose Slope)', '4_80 m den Yuksek Arac giremez (Vehicle higher than 4_80 m cannot enter)',
              'Gevsek Malzemeli Yol (Loose Material Road)', 'Ehli Hayvan Gecebilir (Animal crossing)',
              'Yaya Gecidi (Crosswalk)', 'Duraklamak ve Parketmek Yasaktir (No stopping and parking)',
              'Her Iki Yandan gidiniz (Go from both sides)', 'Sola Surekli Tehlikeli Viraj (Double curve-Right)',
              'Okul Gecidi (School passing)', 'EDS', 'Egimli Yol (Slop)', 'Yol Daralmasi (Road narrowing)',
              'Sagdan Ana Yola Giris (Entry from Right to Main Road)']

# Directory containing test images
test_images_directory = 'C:\\Users\\Hüseyin Erdem Erol\\Desktop\\all_data\\all_data\\test\\images'
labels_directory = 'C:\\Users\\Hüseyin Erdem Erol\Desktop\\all_data\\all_data\\test\\labels'

# Get the list of image files in the directory
image_files = [f for f in os.listdir(test_images_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize evaluation metrics
true_positive = 0
false_positive = 0
false_negative = 0

for image_file in image_files:
    # Read the image
    image_path = os.path.join(test_images_directory, image_file)
    frame = cv2.imread(image_path)

    # Ensure the image is not None
    if frame is not None:
        # Resize the image if needed
        frame = cv2.resize(frame, (640, 480))

        # Get the model predictions
        result = model(frame)

        # Reading bbox, confidence, and class names information
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                Class = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Read the corresponding label
                label_file = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(labels_directory, label_file)

                # Check if the label file exists
                if os.path.exists(label_path):
                    with open(label_path, 'r') as label_file:
                        lines = label_file.readlines()
                        found_match = False  # Flag to check if at least one match is found
                        for line in lines:
                            values = line.strip().split()
                            class_id = int(values[0])
                            x_gt, y_gt, w_gt, h_gt = map(float, values[1:])
                            x1_gt, y1_gt, x2_gt, y2_gt = int((x_gt - w_gt / 2) * frame.shape[1]), int(
                                (y_gt - h_gt / 2) * frame.shape[0]), int((x_gt + w_gt / 2) * frame.shape[1]), int(
                                (y_gt + h_gt / 2) * frame.shape[0])

                            # Compare predicted box with ground truth box
                            iou = calculate_iou((x1, y1, x2, y2), (x1_gt, y1_gt, x2_gt, y2_gt))
                            if iou > 0.3 and Class == class_id:
                                found_match = True
                                true_positive += 1
                                break  # Break the loop if a match is found

                        # If no match is found, increment false positives
                        if not found_match:
                            false_positive += 1

        # Count false negatives
        false_negative += count_false_negatives(labels_directory, image_file, result)

# Calculate precision, recall, and F1 score
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')
