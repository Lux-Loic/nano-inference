import ftplib
from io import BytesIO
import jetson.inference
import jetson.utils
import json
from jtop import jtop
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
import sys
import time
import torch
from torchvision import transforms, models

def get_image(image, width, height):
    # Image transformations
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    # Transform image
    image = trans(image)
    #image = torch.rand(3, 4304, 4304)
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


def load_logger(logger_path):
    # create logger with 'spam_application'
    logger = logging.getLogger('Lux')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger


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
        logger.info("Nano is too hot, taking a 1 minute pause")
        time.sleep(60)


def connect_ftp(FTP_HOST, FTP_USER, FTP_PASS):
    ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
    ftp.cwd('files')
    # force UTF-8 encoding
    ftp.encoding = "utf-8"
    return ftp

logger = load_logger("logs.log")

if __name__ == "__main__":    
    # FTP access 
    FTP_HOST = "10.3.141.1"
    FTP_USER = "pi"
    FTP_PASS = "LuxPi!"
    
    # Stats
    jetson = jtop()
    jetson.start()

    print("----- Lux Ship Detection -----")
    # Paths
    model_path = "./models/model.pth"

    # Tiling
    tile_width = 538
    tile_height = 538

    # Load model
    logger.info("Loading model...")
    print("Loading model...")
    model = load_model(model_path)
    # Send model to device
    model.eval().cuda()
    logger.info("Model Loaded")
    print("Model Loaded")
    
    # Clear Memory
    torch.cuda.empty_cache()

    # Keep track of time
    initial_time = time.time()
    start_time = time.time()
    end_time = time.time()
    execution_time = 3600 # 1 hour
    analysis_time = 60 # 1 minute
    MAX_TEMP = 100
    running = True

    logger.info("Start Analyzing")
    print("Start Analyzing")
    while running:
        # If Nano gets too hot
        heat_check(jetson, MAX_TEMP)
        # For analysis purpose, remove in production
        # Limits execution time to 'execution_time' seconds
        #if (time.time() - initial_time) > execution_time:
        #    running = False
        if (end_time - start_time) >= analysis_time:
            # Clear Memory
            torch.cuda.empty_cache()
            # Retrieve files
            ftp = connect_ftp(FTP_HOST, FTP_USER, FTP_PASS)
            logger.info("Connected to FTP")
            print("Connected to FTP")
            files = ftp.nlst()
            files.sort()
            file_path = files[-1]
            del files
            logger.info("Retrieving " + file_path)
            print("Retrieving " + file_path)
            # Read image
            with open(file_path, "wb") as file:
                flo = BytesIO()
                ftp.retrbinary(f"RETR {file_path}", flo.write)
                flo.seek(0)
                image = Image.open(flo).convert("RGB")
                ftp.quit()
                del ftp
                del file_path
                del flo
                # Get image
                batch = get_image(image, tile_width, tile_height)
                del image
                # Flatten patches
                batch = batch.contiguous().view((batch.size(0) * batch.size(1)), batch.size(2), batch.size(3), batch.size(4))
                # Send to GPU
                batch = batch.cuda()
                # Loop all tiles
                logger.info("Analyzing image...")
                print("Analyzing image...")
                count = 1
                for image in batch:
                    print(count)
                    # Prediction
                    prediction = model(image.unsqueeze(0))
                    count += 1
                del batch
                # Save Prediction
                logger.info("Prediction saved")
                # Reset timer
                start_time = time.time()

        # Set current time
        end_time = time.time()
