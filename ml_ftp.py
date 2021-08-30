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
import yaml


def get_image(image, width, height):
    """
    Converts an image into a tensor and tiles it to smaller patches.
    ---
    image (PIL image): The image in PIL format
    width (int): width of each tile 
    height (int): height of each tile 
    """
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
    img (tensor): the image as a tensor of shape (channels, width, height)
    width (int): the width of every patch
    height (int): the height of every patch
    """
    return img.data.unfold(0, 3, 3).unfold(1, width, height).unfold(2, width, height).squeeze()


def reconstruct(img, tiles):
    """
    Reconstruct an image based on tiles
    ---
    img (tensor): the original image
    tiles (tensor): tiles in the shape (1, rows, cols, channels, width, height)
    """
    return tiles.permute(2, 0, 3, 1, 4).contiguous().view_as(img)


def load_model(model_path):
    """
    Loads the PyTorch model.
    ---
    model_path (string): path to the .pth file
    """
    # Get model from PyTorch
    model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)

    # Load from saved weights
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    # Return model
    return model


def load_logger(logger_path):
    """
    Loads the logger
    ---
    config_path (string): path to the logging file
    """
    # create logger with 'spam_application'
    logger = logging.getLogger('Lux')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger


def load_config(config_path):
    """
    Loads the configuration file
    ---
    config_path (string): path to the configuration file
    """
    config = yaml.safe_load(open(config_path))
    return config


def heat_check(jetson, max_temp):
    """
    Checks if the Nano is getting too hot. If temperature reaches
    the maximum allowed, computation is stopped for a minute.
    ---
    jetson (JTOP): jtop Object
    max_temp (int): the maximum allowed temperature
    """
    if jetson.stats["Temp AO"] >= max_temp or \
            jetson.stats["Temp CPU"] >= max_temp or \
            jetson.stats["Temp GPU"] >= max_temp or \
            jetson.stats["Temp PLL"] >= max_temp or \
            jetson.stats["Temp thermal"] >= max_temp:
        # Take a 60 sec. pause
        logger.info("Nano is too hot, taking a 1 minute pause")
        time.sleep(60)


def connect_ftp(FTP_HOST, FTP_USER, FTP_PASS):
    """
    Connect to the FTP server
    ---
    FTP_HOST (string): host adresse of the server
    FTP_USER (string): username
    FTP_PASS (string): password
    """
    ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
    ftp.cwd('files')
    # force UTF-8 encoding
    ftp.encoding = "utf-8"
    return ftp


def log(logger, message, console=True):
    logger.info(message)
    if console:
        print(message)


logger = load_logger("logs.log")

if __name__ == "__main__":
    # ----------------------------------------
    # Configuration
    # ----------------------------------------
    config = load_config("./config.yml")

    # ----------------------------------------
    # Results
    # ----------------------------------------
    # Create results folder if it doesn't exist
    if not os.path.exists(config["results"]["path"]):
        os.makedirs(config["results"]["path"])

    # Create predictions folder if it doesn't exist
    predictions_path = os.path.join(config["results"]["path"], "./predictions")
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    # ----------------------------------------
    # Jetson Nano
    # ----------------------------------------
    jetson = jtop()
    jetson.start()

    print("----- Lux Ship Detection -----")
    # Paths
    model_path = "./models/model.pth"

    # Load model
    log(logger, "Loading model...")
    model = load_model(config["model"]["torch"]["path"])
    # Send model to device
    model.eval().cuda()
    log(logger, "Model Loaded")

    # Clear Memory
    torch.cuda.empty_cache()

    # Keep track of time
    initial_time = time.time()
    start_time = time.time()
    end_time = time.time()
    running = True

    log(logger, "Start Analyzing")
    while running:
        # If Nano gets too hot
        heat_check(jetson, config["inference"]["max_temperature"])
        # For analysis purpose, remove in production
        # Limits execution time
        if (time.time() - initial_time) > config["inference"]["execution_time"]:
            running = False
        if (end_time - start_time) >= config["inference"]["execution_time"]:
            # Clear Memory
            torch.cuda.empty_cache()
            # Retrieve files
            ftp = connect_ftp(
                config["ftp"]["host"], config["ftp"]["username"], config["ftp"]["password"])
            log(logger, "Connected to FTP")
            files = ftp.nlst()
            files.sort()
            file_path = files[-1]
            del files
            log(logger, f"Retrieving {file_path}")
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
                batch = get_image(
                    image, config["images"]["tile"]["width"], config["images"]["tile"]["height"])
                del image
                # Flatten patches
                batch = batch.contiguous().view((batch.size(0) * batch.size(1)),
                                                batch.size(2), batch.size(3), batch.size(4))
                # Send to GPU
                batch = batch.cuda()
                # Loop all tiles
                log(logger, "Analyzing image...")
                count = 1
                for image in batch:
                    print(count)
                    # Prediction
                    prediction = model(image.unsqueeze(0))
                    # Save prediction to file
                    file_name = file_path.split(
                        '/')[-1].replace(config["images"]["type"], f"_{count}.pth")
                    torch.save(prediction[0], config["results"]
                               ["path"] + "/predictions/" + file_name)
                    # Clear Memory
                    del prediction
                    del file_name
                    torch.cuda.empty_cache()
                    # Increment count
                    count += 1
                # Clear Memory
                del batch
                del count
                del file_path
                # Save Prediction
                log(logger, "Prediction saved", console=False)
                # Reset timer
                start_time = time.time()

        # Set current time
        end_time = time.time()
