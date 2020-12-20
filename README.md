# computervision
This repository consists 3 parts:

# Part 1
Implementing low-level image filtering technique- Gaussian blur to an image.
Procedure: 
1. Load an image file provided on the command line, and decompress it into a numpy array. 
2. Split the input image into 3 channels (R, G, B) 
3. Compute a two-dimensional isotropic Gaussian kernel. 
4. Convolve the Gaussian kernel with each channel of the image. 
5. Save the result.

# Part 2
Using homography matrices to warp planar regions in images. First we rectify a single planar region and then composite one planar region onto another. 
Procedure for Image rectification:
• Take one image with a planar surface.
• Build and compute homography functions.
• Warp the planar regions on the image using homography functions. 
• Crop the image.

Commands:
```
python part1 rectify [input_image_file] [co-ordinates] [output_imag_file]
python part1 rectify [input_image_file] [co-ordinates] --crop [output_image_file]
```


Procedure for Image compositing: 
• Take 2 images one with a planar surface. 
• Compute the homography between two planes in both images. 
• Composite one image onto another image with an image mask.

Commands:
```
python part2 composite [composite_image_file] [input_image_file] [co-ordinates] [composite_mask_image] [output_image_file]
```

# Part 3
To use the iterative Lucas-Kanade algorithm to track an object from one frame to another.

Commands:
```
python part3 --boundingBox [co-ordinates] [image_frame_1] [image_frame_2]
```
