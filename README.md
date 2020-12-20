# computervision
This project consists of 3 parts:


## Part 1
Implementing a low-level image filtering technique- Gaussian blur to an image.


Procedure: 
1. Load an image file provided on the command line, and decompress it into a numpy array. 
2. Split the input image into 3 channels (R, G, B). 
3. Compute a two-dimensional isotropic Gaussian kernel. 
4. Convolve the Gaussian kernel with each channel of the image. 
5. Save the result.


### Command:
```
python part1.py --sigma [sigma_value] --k [kernel_size] [input_image_file] [output_image_file]
```

## Part 2
Using homography matrices to warp planar regions in images. First we rectify a single planar region and then composite one planar region onto another.


Procedure for Image rectification:
1. Take one image with a planar surface.
2. Build and compute homography functions.
3. Warp the planar regions on the image using homography functions. 
4. Crop the image.


### Command:
```
python part2.py rectify [input_image_file] [co-ordinates] [output_image_file]
python part2.py rectify [input_image_file] [co-ordinates] --crop [output_image_file]
```


Procedure for Image compositing:
1. Take 2 images one with a planar surface. 
2. Compute the homography between two planes in both images. 
3. Composite one image onto another image with an image mask.


### Command:
```
python part2.py composite [composite_image_file] [input_image_file] [co-ordinates] [composite_mask_image] [output_image_file]
```


## Part 3
To use the iterative Lucas-Kanade algorithm to track an object from one frame to another.


### Command:
```
python part3.py --boundingBox [co-ordinates] [image_frame_1] [image_frame_2]
```
