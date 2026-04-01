# Stereo Vision Based Distance Estimation

## Overview

This project implements a stereo vision pipeline to estimate object distance from a pair of left and right images. The system uses stereo correspondence to compute a disparity map and then converts disparity into depth using camera calibration parameters. In addition, object detection is applied so that detected classroom objects such as chairs and tables can be localized and their approximate distances can be visualized.

The project demonstrates the full workflow of a stereo vision based measurement system:

1. Read rectified or approximately aligned stereo image pairs
2. Compute disparity between left and right views
3. Convert disparity into depth
4. Detect objects in the scene
5. Estimate distance for each detected object
6. Save visual and numerical outputs

---

## Objective

The goal of this assignment is to understand how stereo vision can be used for 3D distance estimation from two 2D images. The key idea is that an object appears at slightly different horizontal positions in the left and right images. This difference is called **disparity**. Larger disparity generally means the object is closer to the camera, while smaller disparity means the object is farther away.

The relationship used is:

\[
\text{Depth} = \frac{f_x \times B}{d}
\]

where:

- `f_x` = focal length in pixels
- `B` = baseline, or distance between the two camera centers
- `d` = disparity

---

## Features

This project includes the following components:

- Stereo image loading
- Optional image rectification
- Disparity map computation
- Depth map generation
- Object detection on classroom objects
- Estimated object distance display
- Output visualization and result saving
