import pytesseract
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

import csv
   

def alignCard(image, contour):
    #this method is iffy still
    approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True) 

    corners = np.squeeze(approx)

    return corners


def crop_top_percentage(image):
    height, width, _ = image.shape

    perc = 13
    crop_pixels = int(height * perc / 100)
    cropped_image = image[:crop_pixels, :]

    return cropped_image

def crop_set(image):
    height, width, _ = image.shape

    #I am well aware of how stupid this is to hardcode values, but i dont know another way yet
    Topperc = 57
    LeftCrop = int(width * 82 / 100)
    BottomCrop = int(height * 62 / 100)
    RightCrop = int(width * 94 / 100)
    Topcrop_pixels = int(height * Topperc / 100)

    cropped_image = image[Topcrop_pixels:BottomCrop, LeftCrop:RightCrop]

    #print("Crop Error")

    return cropped_image


def checkSet(image, year):

    modelName = "./Models/" + year + "Model.pth"
    csvPath = "./Models/" + year + "Map.csv"

    model = torch.load(modelName)
    model.eval()

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # OG transition from training the Data
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(type(image))
    trueImg = Image.fromarray(image)
    print(type(trueImg))

    img = data_transform(trueImg)


    print(type(img))

    # Add batch dimension to the image to make it work
    img = img.unsqueeze(0)
    img = img.to(device)

    output = model(img)
    print("All of em", output)
    _, predicted = torch.max(output, 1)
    print(predicted.item())
    classList = []
    with open(csvPath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader) 
        for row in csv_reader:
            class_name, idx = row
            classList.append([class_name, int(idx)])

    #print(classList)

    #print(classList[int(predicted.item())][0])

    #we return ONLY the predicted image. If needbe, we could maybe do the CI when comparing multiple?

    return classList[int(predicted.item())][0]











