# Adaptive DualISO HDR Reconstruction

This repository hosts the Adaptive DualISO HDR Reconstruction software, a robust and sensor noise-aware method designed for images captured using the Magic Lantern modification. This software facilitates the reconstruction of a single HDR image and can be adapted for various types of data. For more information about the scientific background and methodology, refer to our [published paper](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-015-0095-0).

## Developer
This program was developed by Saghi Hajisharif at the Visual Computing Laboratory (VCL) at Linköping University.

## Calibration Requirements

Before using this software, careful calibration of the camera is necessary. Please capture the following images:

1. White (Flat-field) and Black images at ISO 100.
2. White (Flat-field) and Black images at ISO 1600 (or any higher ISO of your choice if using ISO 100-1600).
3. White (Flat-field) and Black images with Dual-ISO mode activated on Magic Lantern (ML).

### Notes on Image Capture:
- Black images should be captured in a dark room with the camera lens capped.
- White images should be taken with the lens removed or out of focus. A very flat surface may also be used, ensuring not to saturate the pixels.
- The calibration's accuracy improves with the number of images taken; it is recommended to capture at least 40 images.
- Calibration data is saved as `.mat` files by a MATLAB program.

## Dependencies

- **RawSpeed Library**: Modified for compatibility with Linux. [GitHub Repository](https://github.com/klauspost/rawspeed). An XML file for camera specifications is included in `opensource/rawspeed/cameras.xml`.
- **OpenCV**
- **Eigen**

## How to Run

Execute the program with the following command syntax:
./dualISO input.CR2 out.exr -hmin 0.6 -hmax 5 -hinc 0.1 -fsizex 11 -fsizey 11 -S 0.4 -ICI 0 -M 0 -ALL 1



### Command Line Arguments:
- `-hmin`    Minimum kernel size (default = 1.4)
- `-hmax`    Maximum kernel size (default = 1.4)
- `-hinc`    Kernel size increment (default = 0.1)
- `-fsizex`  Filter size in x dimension (default = 11)
- `-fsizey`  Filter size in y dimension (default = 11)
- `-T`       ICI confidence interval scaling (default = 1.0)
- `-S`       Smoothness parameter (default = 1.0)
- `-ICI`     Applies the ICI rule if set != 0 (default = 1)
- `-M`       Selects the order of the polynomial (0, 1, 2) (default = 2)
- `-ALL`     Adapts RGB channels separately

## Citation

If you use this software in your research, please cite it as follows:

```bibtex
@article{hajisharif2015adaptive,
  title={Adaptive dualISO HDR reconstruction},
  author={Hajisharif, Saghi and Kronander, Joel and Unger, Jonas},
  journal={EURASIP Journal on Image and Video Processing},
  volume={2015},
  pages={1--13},
  year={2015},
  publisher={Springer}
}

