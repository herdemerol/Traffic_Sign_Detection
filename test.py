from ultralytics import YOLO
import cvzone
import cv2
import math
import os

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

        # Getting bbox, confidence, and class names information to work with
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 40:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 0, 0), 1, cv2.LINE_AA)

        # Read the corresponding label
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_directory, label_file)

        # Check if the label file exists
        if os.path.exists(label_path):
            # Read the YOLO format label file
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
                for line in lines:
                    values = line.strip().split()
                    class_id = int(values[0])
                    x, y, w, h = map(float, values[1:])
                    x1, y1, x2, y2 = int((x - w / 2) * frame.shape[1]), int((y - h / 2) * frame.shape[0]), int(
                        (x + w / 2) * frame.shape[1]), int((y + h / 2) * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('frame', frame)
        cv2.waitKey(0)  # Wait for a key press to move to the next image

# Release resources
cv2.destroyAllWindows()
