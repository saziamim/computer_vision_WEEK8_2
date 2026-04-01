# Stereo Report Notes

## Summary
Two classroom images were captured using the same camera from slightly different horizontal positions,
forming a pseudo-stereo uncalibrated setup. Feature correspondences were extracted using SIFT.
The Fundamental Matrix (F) was estimated using RANSAC. Since the camera intrinsics were unknown, an approximate
intrinsic matrix (K) was assumed from the image resolution to compute the Essential Matrix (E). The Rotation Matrix (R)
was recovered from E. The image pair was rectified, disparity was computed using StereoSGBM, and the selected object's
distance was estimated using:

    Z = (f * B) / d

where Z is distance, f is focal length in pixels, B is baseline in cm, and d is disparity in pixels.

## Output Files
- 01_raw_matches.png
- 02_inlier_matches.png
- 03_rectified_left.png
- 04_rectified_right.png
- 05_rectification_check.png
- 06_disparity.png
- 07_annotated_result.png
- matrices_and_results.txt