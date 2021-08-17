import jetson.inference
import jetson.utils
import json
from jtop import jtop
import numpy as np
import os
import pandas as pd
from PIL import Image
import sys
import time
import torch
from torchvision import transforms, models


def get_image(image, width, height):
    '''# Image transformations
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    # Transform image
    image = trans(image)'''
    image = torch.rand(3, 4304, 4304)
    # Split image into tiles
    tiles = tile(image, width, height)
    del image
    # Return
    return tiles


def tile(img, width, height):
    """
    Slices an image into multiple patches
    ---
    img: the image as a tensor of shape (channels, width, height)
    width: the width of every patch
    height: the height of every patch
    """
    return img.data.unfold(0, 3, 3).unfold(1, width, height).unfold(2, width, height).squeeze()


def reconstruct(img, tiles):
    """
    Reconstruct an image based on tiles
    ---
    img: the original image
    tiles: tiles in the shape (1, rows, cols, channels, width, height)
    """
    return tiles.permute(2, 0, 3, 1, 4).contiguous().view_as(img)


def load_model(model_path):
    # Get model from PyTorch
    model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)

    # Load from saved weights
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    # Return model
    return model

def load_model2(model_path):
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True    
    )
    return model


def heat_check(jetson, max_temp):
    """
    Checks if the Nano is getting too hot. If temperature reaches
    the maximum allowed, computation is stopped for a minute.
    ---
    max_temp: the maximum allowed temperature
    """       
    if  jetson.stats["Temp AO"] >= max_temp or \
        jetson.stats["Temp CPU"] >= max_temp or \
        jetson.stats["Temp GPU"] >= max_temp or \
        jetson.stats["Temp PLL"] >= max_temp or \
        jetson.stats["Temp thermal"] >= max_temp:
        # Take a 60 sec. pause
        print("Nano is too hot, taking a 1 minute pause")
        time.sleep(60)


if __name__ == "__main__":
    model = load_model("./models/ssdlite320_mobilenet_v3_large.pth")
    model.eval().cuda()
    print("Model Loaded")
    torch.cuda.empty_cache()

    # Get image
    image = Image.open("./images/image.tif").convert("RGB")
    trans = transforms.Compose([transforms.ToTensor()])
    image = trans(image)
    print(image.shape)

    # Predict
    start_time = time.time()
    preds = model(image.unsqueeze(0).cuda())
    print("--- %s seconds ---" % (time.time() - start_time))
