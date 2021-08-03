import jetson.inference
import jetson.utils
from jtop import jtop
import numpy as np
import os
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
        jetson.stats["Temp thermal"] >= max_temp:
        # Take a 60 sec. pause
        print("Nano is too hot, taking a 1 minute pause")
        time.sleep(60)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        images_folder = args[0]
        if not os.path.isdir(images_folder):
            sys.exit("Invalid image folder path")
    else:
        sys.exit("Invalid image folder path")

    jetson = jtop()
    jetson.start()

    print("----- Lux Ship Detection -----")
    # Clear Memory
    torch.cuda.empty_cache()
    # Paths
    model_path = "./models/ssdlite320_mobilenet_v3_large.pth"
    #model_path = "./models/shufflenet.pth"

    # Tiling
    tile_width = 269
    tile_height = 269

    # Load model
    print("Loading model...")
    model = load_model(model_path)
    # Send model to device
    model.eval().cuda()
    print("Model Loaded")

    # Keep track of time
    initial_time = time.time()
    start_time = time.time()
    end_time = time.time()
    execution_time = 3600 # 1 hour
    analysis_time = 60 # 1 minute
    MAX_TEMP = 100
    running = True

    print("Start Analyzing")
    while running:
        # If Nano gets too hot
        heat_check(jetson, MAX_TEMP)
        # For analysis purpose, remove in production
        # Limits execution time to 'execution_time' seconds
        if (time.time() - initial_time) > execution_time:
            running = False
        if (end_time - start_time) >= analysis_time:
            # Clear Memory
            torch.cuda.empty_cache()
            # Retrieve files
            files = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder)]
            files = sorted(files)
            file_path = files[-1]
            del files
            print("Analyzing " + file_path)
            # Read image
            image = Image.open(file_path).convert('RGB')
            del file_path
            # Get image
            batch = get_image(image, tile_width, tile_height)
            del image
            print(batch.shape)
            # Send to GPU
            #batch = batch.cuda()
            # Flatten patches
            batch = batch.contiguous().view((batch.size(0) * batch.size(1)), batch.size(2), batch.size(3), batch.size(4))
            batch = batch.cuda()
            # Loop all tiles
            count = 1
            for image in batch:
                print(count)
                # Prediction
                #prediction = model(image.unsqueeze(0).cuda())
                prediction = model(image.unsqueeze(0))
                count += 1
            del batch
            # Save Prediction
            print("Prediction saved")
            # Reset timer
            start_time = time.time()

        # Set current time
        end_time = time.time()
        #print(end_time - start_time)
