import torch
from torchvision.transforms import v2
import csv


def create_encoding_dict(csv_file, delimiter):
    encoding_dict = {}

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=delimiter)
        for row in csv_reader:
            label = row['Label']
            index = int(row['Encoding'])
            encoding_dict[index] = label

    return encoding_dict


def create_info_dict(csv_file, delimiter):
    info_dict = {}

    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        next(csv_reader)

        for row in csv_reader:
            label, info, link = row
            info_dict[label] = [info, link]

    return info_dict


def classify_image(model, encoding_dict, image):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(384, 384), antialias=True),

        v2.ToDtype(torch.float32, scale=True),
        # mean and std from train set resized to 384 x 384
        v2.Normalize(mean=[0.41906729340553284, 0.4273952543735504, 0.4206131398677826],
                     std=[0.2959694266319275, 0.2940460443496704, 0.3156360685825348])
    ])

    transformed_image = transform(image)

    output = model(transformed_image.unsqueeze(0))  # unsqueeze to add batch dimension

    probability_distribution = torch.nn.functional.softmax(output[0], dim=0)
    probability, predicted_class_index = torch.max(probability_distribution, 0)
    probability = probability.item()
    predicted_class_index = predicted_class_index.item()
    predicted_class = encoding_dict[predicted_class_index]

    return predicted_class, probability
