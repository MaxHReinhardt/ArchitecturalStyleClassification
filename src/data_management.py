import os
import random
import csv
import shutil


def split_data(source_folder, output_folder, train_frac=0.6, test_frac=0.2, val_frac=0.2, seed=1):
    """
    Splits data in train/validation/test set and creates corresponding annotation files. Further, classes are encoded
    as unique integers and encodings are displayed in a mapping file. Works with the file structure of the following
    dataset: https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset
    """

    # Create output folders for train/validation/test set if they don't exist
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    val_folder = os.path.join(output_folder, 'validation')

    for folder in [train_folder, test_folder, val_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get the list of class labels from folder names and encode them
    classes = os.listdir(source_folder)
    class_encoding = {label: i for i, label in enumerate(classes)}

    # Write class encoding to a CSV file
    encoding_file_path = os.path.join(output_folder, 'class_encoding.csv')
    with open(encoding_file_path, 'w', newline='') as encoding_file:
        encoding_writer = csv.writer(encoding_file)
        encoding_writer.writerow(['Label', 'Encoding'])
        for label, encoding in class_encoding.items():
            encoding_writer.writerow([label, encoding])

    for label in classes:
        # Create subfolders for classes in output folders
        os.makedirs(os.path.join(train_folder, label), exist_ok=True)
        os.makedirs(os.path.join(test_folder, label), exist_ok=True)
        os.makedirs(os.path.join(val_folder, label), exist_ok=True)

        # Get the list of images for each class
        images = os.listdir(os.path.join(source_folder, label))

        # Shuffle the images randomly
        random.seed(seed)
        random.shuffle(images)

        # Calculate the split indices (Splitting for each class preserves class balances.)
        train_split = int(train_frac * len(images))
        test_split = int((train_frac + test_frac) * len(images))

        # Split into train/test/validation set
        train_images = images[:train_split]
        test_images = images[train_split:test_split]
        val_images = images[test_split:]

        # Move images to corresponding folders and rename
        move_and_rename(source_folder, train_images, label, train_folder)
        move_and_rename(source_folder, test_images, label, test_folder)
        move_and_rename(source_folder, val_images, label, val_folder)

    # Create annotation files in CSV format for train/test/validation set
    create_annotation_file_csv(train_folder, os.path.join(output_folder, 'train_annotation.csv'), class_encoding)
    create_annotation_file_csv(test_folder, os.path.join(output_folder, 'test_annotation.csv'), class_encoding)
    create_annotation_file_csv(val_folder, os.path.join(output_folder, 'validation_annotation.csv'), class_encoding)

def move_and_rename(source_folder, images, label, output_folder):
    for i, image in enumerate(images):
        old_path = os.path.join(source_folder, label, image)
        new_name = f"{label}_{i + 1}.jpg"
        new_path = os.path.join(output_folder, label, new_name)
        shutil.copy(old_path, new_path)

def create_annotation_file_csv(folder, annotation_file_path, class_encoding):
    with open(annotation_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image Path', 'Label Encoding'])

        for label in os.listdir(folder):
            encoding = class_encoding[label]
            for image in os.listdir(os.path.join(folder, label)):
                image_path = os.path.join(folder, label, image)
                csv_writer.writerow([image_path, encoding])


if __name__ == "__main__":
    source_folder = 'data/archive/architectural-styles-dataset'
    output_folder = 'data/dataset'
    split_data(source_folder, output_folder, train_frac=0.6, test_frac=0.2, val_frac=0.2, seed=1)
