import os
import zipfile

def unzip_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_data_paths(base_dir):
    classes = os.listdir(base_dir)
    data = []
    labels = []
    for label in classes:
        class_dir = os.path.join(base_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                data.append(os.path.join(class_dir, img_file))
                labels.append(label)
    return data, labels
