# Repository Notice

Code For "Towards Specular Highlight Removal Through Diffusion Model" (PRCV-24)

## Requirement
Python 3.7
Pytorch 1.7
CUDA 11.1
pip install -r requirements.txt


## Train
python sr.py -p train -c config/train.json

## Test
python sr.py -p train -c config/test.json
