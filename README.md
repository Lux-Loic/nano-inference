# Nano Inference

This repository contains scripts meant to be executed during launches.

### Machine Learning
The `ml.py` script starts the object detection system.

It takes a single argument which is the path to the folder containing the images.

ex: `python ml.py ./path/to/images`

### Statistics
The `stats.py` script saves all the metrics available from the Nano (heat, CPU usage, etc.)

It generates a CSV file.

ex: `python stats.py`
