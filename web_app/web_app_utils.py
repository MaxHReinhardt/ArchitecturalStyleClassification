import torch
from torchvision.transforms import v2
import csv


def create_encoding_dict(csv_file):
    encoding_dict = {}

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            label = row['Label']
            index = int(row['Encoding'])
            encoding_dict[index] = label

    return encoding_dict


def classify_image(model, encoding_dict, image):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(384, 384), antialias=True),

        v2.ToDtype(torch.float32, scale=True),
        # mean and std from train set resized to 384 x 384
        v2.Normalize(mean=[0.41999170184135437, 0.4282282888889313, 0.42107370495796204],
                     std=[0.29552504420280457, 0.2937185764312744, 0.3150707483291626])
    ])

    transformed_image = transform(image)

    output = model(transformed_image.unsqueeze(0))  # unsqueeze to add batch dimension

    probability_distribution = torch.nn.functional.softmax(output[0], dim=0)
    probability, predicted_class_index = torch.max(probability_distribution, 0)
    probability = probability.item()
    predicted_class_index = predicted_class_index.item()
    predicted_class = encoding_dict[predicted_class_index]

    return predicted_class, probability
