
# Overview

Identifies targets in a CPTV video file using a pre-trained classifier.

# Setup

Create a virtual environment and install the necessary prerequisites 

`pip install -r requirements.txt`

Optionally install GPU support for tensorflow (note this requires additional [setup](https://www.tensorflow.org/install/)

`pip install tensorflow-gpu`  

MPEG4 output requires FFMPEG to be installed which can be found [here](https://www.ffmpeg.org/).  

On windows the installation path will need to be added to the system path. 


# Usage

`python classify.py [cptv filename] -p`

This will generate a text file listing the animals identified, and create an MPEG file.   `

