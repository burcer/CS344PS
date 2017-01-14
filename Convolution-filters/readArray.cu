#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "timer.h"
//#include "reference_calc.h"
//#include "reference_calc.cpp"
#include "st2.cu"
//#include <algorithm>
//#include <cassert>
//#include <stdlib.h>
using namespace std;

void readCSV(istream &input, vector< vector<string> > &output)
{
   string csvLine;
    // read every line from the stream
    while( getline(input, csvLine) )
    {
            istringstream csvStream(csvLine);
           vector<string> csvColumn;
            string csvElement;
            // read every element from the line that is seperated by commas
            // and put it into the vector or strings
            while( getline(csvStream, csvElement, ',') )
            {
                    csvColumn.push_back(csvElement);
            }
            output.push_back(csvColumn);
    }
}


void get_rgba(uchar4* image_rgba)//(uchar4 **d_image_rgba, uchar4 **h_image_rgba){
{
  int rows = 1080;//720;//313;
  int cols = 1920;//960;//557;
  int ch = 4;
  

  
  /////uchar4 *image_rgba;
  //*image_rgba = malloc(rows*cols*sizeof(uchar4));
  /////image_rgba = (uchar4*)malloc(rows*cols*sizeof(uchar4));
  //cudaMallocHost((void **)&image_rgba, rows*cols*sizeof(uchar4));
////////  cudaMallocHost(h_image_rgba, rows*cols*sizeof(uchar4));
////////  cudaMalloc(d_image_rgba, rows*cols*sizeof(uchar4));
  int *arr;
  arr = (int*) malloc(rows*cols*ch*sizeof(int));
  string a;
  fstream file("file2.txt", ios::in);
  if(!file.is_open()){
    cout << "File not found!\n";
            //return 1;
  }
    // typedef to save typing for the following object
  typedef vector< vector<string> > csvVector;
  csvVector csvData;
  readCSV(file, csvData);
  int index=0;
  int x;
  for(csvVector::iterator i = csvData.begin(); i != csvData.end(); ++i)
  {
    for(vector<string>::iterator j = i->begin(); j != i->end(); ++j)
    {
      a=*j;
      //cout << a << " ";
      stringstream convert(a);
      convert>>x;
      arr[index] = x;
      index++;   
    }
  
  }
  cout << index<< "\n";
//  for(int j=0; j<7; j++){
//    cout << arr[j]<< " ";
//  }
//  cout << "\n";
  
//  for (int i = 0; i<rows*cols; i+=4){
//    image_rgba[i].x =  arr[i];
//    image_rgba[i].y =  arr[i+rows*cols];
//    image_rgba[i].z =  arr[i+2*rows*cols];
//    image_rgba[i].w =  arr[i+3*rows*cols];
//  }  
//  for (int i = 0; i<rows*cols; i++){
//    image_rgba[i].x =  arr[4*i];
//    image_rgba[i].y =  arr[4*i+1];
//    image_rgba[i].z =  arr[4*i+2];
//    image_rgba[i].w =  arr[4*i+3];
//  }  
  for (int i = 0; i<rows*cols; i++){
    image_rgba[i].x =  arr[4*i];
    image_rgba[i].y =  arr[4*i+1];
    image_rgba[i].z =  arr[4*i+2];
    image_rgba[i].w =  arr[4*i+3];
    //cout<<i<< "/ ";
  }  
  //cout << arr[rows*cols+3] << " ";
  cout<<"3\n";
  
//////  
//////  cout << (uint)image_rgba[0].x << " ";
//////  cout << (uint)image_rgba[0].y << " ";
//////  cout << (uint)image_rgba[0].z << " ";
//////  cout << (uint)image_rgba[rows*cols-1].x << " ";
//////  cout << (uint)image_rgba[rows*cols-1].y << " ";
//////  cout << (uint)image_rgba[rows*cols-1].z << " \n";
//////  checkCudaErrors(cudaMemcpy(*h_image_rgba,image_rgba, rows*cols*sizeof(uchar4),cudaMemcpyHostToHost));
//////  cudaDeviceSynchronize();
//////  checkCudaErrors(cudaMemcpy(*d_image_rgba,*h_image_rgba, rows*cols*sizeof(uchar4),cudaMemcpyHostToDevice));
//////  //cout << (uint)(*d_image_rgba)[rows*cols-1].z << " \n";
//////  //free(&arr);
////// //free(image_rgba);free(arr);
 
 //return image_rgba;
}



//referenceCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
//                          size_t numRows, size_t numCols,
//                          const float* const filter, const int filterWidth)


void fromCharToText(uchar4* h_grey, size_t rows, size_t cols){
  ofstream outputFile("grey7.txt");
  for(int i=0; i<cols; i++){
    for(int j=0; j<rows; j++){
      if((i+j)<rows+cols-2){
        outputFile << (uint) h_grey[i*rows+j].x << "," <<(uint) h_grey[i*rows+j].y << "," <<(uint) h_grey[i*rows+j].z<< "," <<(uint) h_grey[i*rows+j].w<<"," ;}
        else outputFile << (uint) h_grey[i*rows+j].x << "," <<(uint) h_grey[i*rows+j].y << "," <<(uint) h_grey[i*rows+j].z<< "," <<(uint) h_grey[i*rows+j].w ;
    }
  }
  
}

__global__
void gpuCalculation(uchar4* d_rgba, unsigned char* d_grey){

  //int index = blockDim.x* blockIdx.y + threadIdx.x;
  int index = blockIdx.x;
  uchar4 rgba = d_rgba[index];
  d_grey[index] = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;

}


int main(){

  cout<<"ali\n";
  size_t rows = 1080;
  size_t cols = 1920;
  int ch = 4;

  uchar4 *d_rgba;
  uchar4 *h_rgba, *h_rgba2;
  
 h_rgba2 = (uchar4*)malloc(rows*cols*sizeof(uchar4));
  //cudaMallocHost(&h_rgba2, rows*cols*sizeof(uchar4));
  
  uchar4 *h_out_rgba;
  uchar4 *d_rgba2;
  uchar4 *d_out_rgba;
  cout<<"before calc\n";
  checkCudaErrors(cudaMalloc((void **)&d_out_rgba, rows*cols*sizeof(uchar4)));
  checkCudaErrors(cudaMemset(d_out_rgba, 0, rows*cols*sizeof(uchar4)));

  h_out_rgba = (uchar4*)malloc(rows*cols*sizeof(uchar4));

  memset(h_out_rgba, 0, rows*cols*sizeof(uchar4));
  cout<<"before calc\n";
  //get_rgba(&d_rgba, &h_rgba);
  
  
    int *arr;
  arr = (int*) malloc(rows*cols*ch*sizeof(int));
  string a;
  fstream file("file2.txt", ios::in);
  if(!file.is_open()){
    cout << "File not found!\n";
            //return 1;
  }
    // typedef to save typing for the following object
  typedef vector< vector<string> > csvVector;
  csvVector csvData;
  readCSV(file, csvData);
  int index=0;
  int x;
  for(csvVector::iterator i = csvData.begin(); i != csvData.end(); ++i)
  {
    for(vector<string>::iterator j = i->begin(); j != i->end(); ++j)
    {
      a=*j;
      //cout << a << " ";
      stringstream convert(a);
      convert>>x;
      arr[index] = x;
      index++;   
    }
  
  }
  cout << index<< "\n";
//  for(int j=0; j<7; j++){
//    cout << arr[j]<< " ";
//  }
//  cout << "\n";
  
//  for (int i = 0; i<rows*cols; i+=4){
//    image_rgba[i].x =  arr[i];
//    image_rgba[i].y =  arr[i+rows*cols];
//    image_rgba[i].z =  arr[i+2*rows*cols];
//    image_rgba[i].w =  arr[i+3*rows*cols];
//  }  
//  for (int i = 0; i<rows*cols; i++){
//    image_rgba[i].x =  arr[4*i];
//    image_rgba[i].y =  arr[4*i+1];
//    image_rgba[i].z =  arr[4*i+2];
//    image_rgba[i].w =  arr[4*i+3];
//  }  
  for (int i = 0; i<rows*cols; i++){
    h_rgba2[i].x =  arr[4*i];
    h_rgba2[i].y =  arr[4*i+1];
    h_rgba2[i].z =  arr[4*i+2];
    h_rgba2[i].w =  arr[4*i+3];
    //cout<<i<< "/ ";
  }  
  
  
  
  
  
  
  
  
  ///get_rgba(h_rgba2);//(&d_rgba, &h_rgba);
  checkCudaErrors(cudaMalloc(&d_rgba2, rows * cols * sizeof(uchar4)));
  checkCudaErrors(cudaMemcpy(d_rgba2, h_rgba2, rows * cols * sizeof(uchar4), cudaMemcpyHostToDevice));

  
//  cout<<"00\n";
//     uchar4 rgba = h_rgba2[0];
//     cout<<"1\n";
//    unsigned char red;
//    cout<<"2\n";
//    red   = rgba.x;
//    cout<<(uint)red <<"\n";

  
  cout<<"before calc\n";
  //float filter[9] = {0, 0.17, 0, 0.17,0.32,0.17,0,0.17,0};
  //float filter[9] = {0,-1,0,-1,5,-1,0,-1,0};
 
  cout<<"before ref\n";
  //referenceCalculation(h_rgba2, h_out_rgba, rows, cols,filter,3);
  //dadssdint filterWidth=3;
  
    //now create the filter that they will use
  const int blurKernelWidth = 13;
  const float blurKernelSigma = 2.;

  int filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  float *filter = new float[blurKernelWidth * blurKernelWidth];

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }

  
  unsigned char *d_redBlurred;
  unsigned char *d_greenBlurred;
  unsigned char *d_blueBlurred;
  checkCudaErrors(cudaMalloc((void **)&d_redBlurred, rows*cols*sizeof(unsigned char)));
  checkCudaErrors(cudaMemset(d_redBlurred, 0, rows*cols*sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void **)&d_greenBlurred, rows*cols*sizeof(unsigned char)));
  checkCudaErrors(cudaMemset(d_greenBlurred, 0, rows*cols*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc((void **)&d_blueBlurred, rows*cols*sizeof(unsigned char)));
  checkCudaErrors(cudaMemset(d_blueBlurred, 0, rows*cols*sizeof(unsigned char)));  
  allocateMemoryAndCopyToGPU(rows, cols, filter,  filterWidth);
    GpuTimer timer;
  timer.Start(); 
  //your_gaussian_blur(h_rgba2, d_rgba2, d_out_rgba, rows,cols,d_redBlurred,d_greenBlurred, d_blueBlurred,filterWidth);
  referenceCalculation(h_rgba2, h_out_rgba,rows, cols,filter,filterWidth);
    cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
  timer.Stop();
  float elapsed = timer.Elapsed();
  cout<< elapsed << "\n ";
  
//  GpuTimer timer;
//  timer.Start();  
//  //referenceCalculation(h_rgba, h_grey,rows, cols);
//  const dim3 blockSize(1, 1, 1);
//  const dim3 gridSize(313*557,1 , 1);
//  gpuCalculation<<<gridSize,blockSize>>>(d_rgba, d_grey);
//  cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
//  timer.Stop();
//  float elapsed = timer.Elapsed();
//  cout<< elapsed << " ";
  
  //cudaMemcpy(h_out_rgba, d_out_rgba, rows*cols*sizeof(uchar4), cudaMemcpyDeviceToHost);
//  

//cout<<(uint) h_out_rgba[100* cols+ 100].x<<"ddffd\n";
 fromCharToText(h_out_rgba,rows, cols);
  
//  cudaFree(d_out_rgba);cudaFree(d_rgba);cudaFree(h_rgba); free(h_out_rgba); 
//  

  return 0;

}





//void fileToImage(uchar4 **rgbaImage, const std::string &filename, int rows, int cols, int ch){

//  int *stream;
//  stream = (int*) malloc()

//}

