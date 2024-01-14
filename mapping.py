import os


# Define a dictionary to map the old values to the new values
mapping = {
    41: 40,
    40: 39,
    39: 38,
    38: 37,
    37: 36,
    36: 35,
    35: 34,
    34: 33,
    33: 32,
    32: 31,
    31: 30,
    30: 29,
    29: 28,
    28: 27,
    27: 26,
    26: 25,
    25: 24,
    24: 23,
    23: 22,
    22: 21,
    21: 20,
    20: 19,
    19: 18,
    18: 17,
    17: 16,
    16: 15,
    15: 14,
    14: 13,
    13: 12,
    12: 11,
    11: 10,
    10: 9,
    9: 8,
    8: 7,
    7: 6,
    6: 5,
    5: 4,
    4: 3,
    3: 2,
    2: 1,
    1: 0
}
folder_path = 'C:\\Users\\Hüseyin Erdem Erol\\Desktop\\all_data\\valid\\labels'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # Read the content of the current file
        with open(file_path, 'r') as file:
            yolo_lines = file.readlines()

        # Process the content of the current file
        new_yolo_lines = []
        for line in yolo_lines:
            parts = line.split()
            if len(parts) > 0:
                parts[0] = str(mapping.get(int(parts[0]), int(parts[0])))
                new_yolo_lines.append(' '.join(parts))

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines('\n'.join(new_yolo_lines))

print("Modification completed for all YOLO files in the folder.")
