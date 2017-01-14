# CS344PS
Parallel computing class assignments: Image processing on GPU with CUDA/C++
.cu files are compiled with nvcc. I use CUDA8.0, with Quadro K2200 GPU. OS is Ubuntu 14.04

.cu files to compile:
readArray.cu for colorToGray

readArray.cu for Convolution-filters

HW3.cu for HDR-Histogram calculations-ToneMapping


Normally, assignments required OpenCV. But I wrote Python code to convert images to datafiles with integers; and input those files to .cu files. You should be good to use this by compiling the files above.
