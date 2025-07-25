# Seafloor Sediment Segmentation with U-Net

This project implements a deep learning pipeline for semantic segmentation of seafloor images using a U-Net architecture with a ResNet backbone. The model predicts 17 classes of sediment and substrate types based on RGB images and accompanying bathymetry data.

## Features

- U-Net with configurable backbone (`resnet18`, `resnet50`, `vgg16`, etc.)
- 17 sediment classes with custom color mapping
- Preprocessing and normalization for bathymetry input
- Training with class weights for imbalance handling
- Evaluation using Dice coefficient and Intersection over Union (IoU)
- End-to-end training and prediction pipeline in Google Colab

## Dataset

Images and masks must be organized in the following structure within Google Drive:

```
MyDrive/
├── seafloor_images/       # Directory with TIFF image patches
└── seafloor_masks/        # Directory with PNG segmentation masks
```

Ensure image names align between `seafloor_images` and `seafloor_masks` for correct pairing.

## Installation

This notebook runs in Google Colab. To install required dependencies:

```bash
!pip install rasterio tifffile scikit-image imutils
```

## Label Information

There are 17 sediment/substrate classes, each mapped to a unique RGB color in `VOC_COLORMAP` and `VOC_CLASSES`.

Example:

```python
VOC_CLASSES = ['Sand', 'Gravel', 'Rock', 'Mud', 'Sand-gravel', ...]
VOC_COLORMAP = [[255,255,0], [128,0,128], [128,128,128], [0,0,255], ...]
```

The label masks use these RGB values to encode each class.

## Preprocessing

- Input images resized to 224x224
- Bathymetry channel is normalized based on min/max values observed
- Masks are converted from RGB to integer class maps using `voc_label_indices()`
- Optional one-hot encoding available for advanced loss functions

## Model

A modified U-Net with optional pretrained encoder backbones. Configurable parameters include:

- `backbone_name`: ResNet, VGG, or DenseNet variants
- `decoder_filters`: Tuple defining decoder channel sizes
- `input_channels`: RGB + bathymetry (typically 3)
- `parametric_upsampling`: Use of transposed convolution vs interpolation

## Training

```python
NUM_EPOCHS = 50
BATCH_SIZE = 32
INIT_LR = 0.0001
DEVICE = "cuda" if available
```

- Uses `CrossEntropyLoss` with class weights computed via `sklearn.utils.class_weight`
- Learning rate scheduler: `CosineAnnealingLR`
- Loss curves are plotted and saved to `plot.png`
- Model checkpoints saved to `seafloor-segment.pth`

## Evaluation

After training, the model is evaluated on a held-out test set:

- Dice Coefficient and IoU computed per image
- Qualitative plots of predictions vs. ground truth
- Results are visualized with RGB overlays

## Inference

Use the `make_predictions()` function to visualize the model's output on unseen images. Predictions are colorized using the VOC colormap for easier interpretation.

## Notes

- The pipeline filters and removes duplicate images with filenames containing `(1)`
- ResNet backbones use torchvision pretrained weights
- Models expect `.tif` images and `.png` masks
- BatchNorm and ReLU activations used throughout

## Output

- Training loss drops steadily across epochs
- Final test IoU ~0.74
- Qualitative examples show strong alignment with true masks
