# Image Segmentation 

using keras-segmenation mobilenet.

## Prepare dataset 

* create folders frames, Labels, FrPNG, Masks, AugFr, AugMasks, ResizedFr, ResizedMasks, CMasks, TestFr, TestM,  outputs.

### frames extractor
* Get a video of the object (Summit XL) you want to train the detector on.
* Split this video into frames using 0.frames_extractor.ipynb.

### labeling
* Label some of the frames of the video (at least 200) and save the annotations in folder 'Labels' in json format. Use labelme tool -> https://github.com/wkentaro/labelme
* Use 0a.syncframes-labels.ipynb to move frames that have annotations,  from folder frames to folder FrPNG.
* Also adds a 10% of frames that the object doesn't appear.

### masking
* Run 1.TakeMasks.ipynb that creates a mask for each frame. 
* Also adds black masks for the frames that object does not appear.

### augmenting
* Run 2.Augment.ipynb that augments the frames and their masks and save them to folders AugFr and AugMasks.
* Also copies original FrPNG and Masks into those augmented folders.

### resize
* Run 3.Resize to resize frames and masks to a lower resolution.
* Saves resized to ResizedFr, ResizedMasks.

### classMasks
* Run 4.ClassMasks. Transform the masks from values 0 and 255 to 0 and 1 so that the detector can assume 2 classes.
* Also moves 10% of frames and masks to TestFr and TestM.

## Train dataset

* using https://github.com/divamgupta/image-segmentation-keras
* use the virtual environment created [here](https://github.com/AndreasMit/stalker/blob/main/build.md)
* Run 5.ModelTrain or train.py that trains on the dataset for a number of epochs.
* Use 6.Predict or predict.py that predicts the masks of frames and saves them to folder 'outputs'.

## Display frames with masks

* At every point of the pipeline above you can check whether the masks apply correctly to the frames using DisplayFrMasks.ipynb. Edit the paths depending on what you are testing.



