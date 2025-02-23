
# VR Assignment 1: Coin Detection & Image Stitching

#### Overview 

This project consists of two parts:

- Coin Detection, Segmentation, and Counting: Uses computer vision techniques to detect, segment, and count Indian coins from an image.
- Panorama Creation: Stitches multiple overlapping images into a seamless panorama.

### Requirements 

```bash
  git clone https://github.com/varnit-mittal/VR_Assignment1_VarnitMittal_IMT2022025.git
  cd VR_Assignment1_VarnitMittal_IMT2022025
  pip install -r requirements.txt
```

### Part 1: Coin Detection & Segmentation

#### Features:
- Detects coins using edge detection.
- Segments each detected coin using contour analysis.
- Counts and displays the total number of coins.

#### Running the code 

```bash
  cd Part1
  python part1.py
```

Now there are 4 input images in input folder and are named as `imagex.jpg` where x is the number of the image, you need to enter which image you want to process through this code and if you want to add your own image, add your image to input folder using the same naming convention.

#### Output:

- Processed images are saved in the `output` folder.
- Outlined detected coins.
- Segmented images of each coin.
- Displayed total coin count.

### Part 2: Image Stitching

#### Features:
- Detects key points using SIFT.
- Matches key points between overlapping images.
- Uses homography to align and stitch images.

#### Running the code 

```bash
  cd Part2
  python part2.py
```

There are 5 input directories right now, you need to enter the name of the input directory and the name of the output directory (if you haven't created an output directory, then it will automatically be created.)

#### Output:
- Stitched panorama saved as `panorama.jpg` in the output folder.
- Intermediate images (`stitched_1.jpg`, etc.) showing progress.

### Obervations and Results
- The coin detection method successfully detects and segments coins with high accuracy.
- The image stitching method performs well on images with sufficient overlap and distinct features.
## Documentation

[Documentation](https://github.com/varnit-mittal/VR_Assignment1_VarnitMittal_IMT2022025/blob/main/IMT2022025_VarnitMittal.pdf)

