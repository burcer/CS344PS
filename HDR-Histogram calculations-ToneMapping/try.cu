#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "timer.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <cmath>
using namespace std;

// CPU function to generate a vector of random integers
void random_ints (float *x, int n) {
  for (int i = 0; i < n; i++)
  *(x+i) = (float) (rand() % 32); // random number between 0 and 9999
}

__global__
void calculate_max(const size_t size, float *array, float *intermed){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int bound = blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
 int flag = blockDim.x%2;
 int comp;
 if (blockDim.x ==1){flag =0;}

  for (int s = (blockDim.x +flag)/2; s>=1; s=(s+flag)/2){  
    comp = s-flag;   
    if(tid < comp){
      array[id] = min(array[id+s],array[id]) ;  

    }  
    flag = s%2; 
    if(s==1) {flag=0;}
    __syncthreads();
  }
  intermed[blockIdx.x] = array[bound];
}

__global__
void calculate_max2(const size_t size, const float *array, float *intermed){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int bound = blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float cache[];
  cache[tid] = array[id]; 
  __syncthreads();


  for (int s = (blockDim.x +1)/2; s>=1; s=s/2){  
       
    if(tid < s){
      cache[tid] = max(cache[tid+s],cache[tid]) ;  

    }  

    __syncthreads();
  }
  intermed[blockIdx.x] = cache[0];
}
__global__
void calculate_max3(size_t size, float *array, float *intermed){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int bound = blockDim.x * blockIdx.x;
  int tid = threadIdx.x; 
  __syncthreads();


  for (int s = (blockDim.x +1)/2; s>=1; s=s/2){  
       
    if(tid < s){
      array[id] = max(array[id+s],array[id]) ; 

    }  

    __syncthreads();
  }
  intermed[blockIdx.x] = array[bound];
}


__global__
void my_histo(int* histo, float* logLum, float minLogLum, float logLumRange, int numBins){

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int bin_pre =(int) ((logLum[id] - minLogLum)/logLumRange * numBins);
  int bin = min(bin_pre , numBins-1);
  atomicAdd(&(histo[bin]),1);

}

__global__
void my_histo2(int perThread, int* histo, int size, float* logLum, float minLogLum, float logLumRange, const int numBins){
 
   __shared__ int collectiveHisto[4*1024];
//  int local_histo[numBins];
  //memset(collectiveHisto, 0, 4*1024*sizeof(int));
  
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  int blockdim = blockDim.x;

  
  for (int i =0; i < numBins; i++){collectiveHisto[numBins*tid+i]=0;}
   __syncthreads();
  for(int j= id*perThread; j < (id+1)*perThread; j++){
      int bin_pre =(int) ((logLum[j] - minLogLum)/logLumRange * numBins);
      int bin = min(bin_pre , numBins-1);
      if (bin>=0){
      collectiveHisto[blockdim*bin + tid]+=1;}
      //atomicAdd(&(histo[bin]),1); 
  }
  __syncthreads();
    
   for(int j=0; j<numBins; j++){ 
        
   for (int s = (1+blockDim.x)/2; s>=1; s=s/2){  
       
    if(tid < s){
      collectiveHisto[blockdim*j+tid] += collectiveHisto[blockdim*j+tid+s] ; 

    }  

    __syncthreads();
  }
  histo[j] = collectiveHisto[blockdim*j];
  
  }
}

__global__
void my_histo3(int perThread, int* histo, int size, float* logLum, float minLogLum, float logLumRange, const int numBins){
 
   __shared__ int collectiveHisto[4*1024];
//  int local_histo[numBins];
  //memset(collectiveHisto, 0, 4*1024*sizeof(int));
  
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  int blockdim = blockDim.x;
  
  
  
  
  for (int i =0; i < numBins; i++){collectiveHisto[numBins*tid+i]=0;}
   __syncthreads();
  for(int j= id*perThread; j < (id+1)*perThread; j++){
      int bin_pre =(int) ((logLum[j] - minLogLum)/logLumRange * numBins);
      int bin = min(bin_pre , numBins-1);
      collectiveHisto[blockdim*bin + tid]+=1;
      //atomicAdd(&(histo[bin]),1); 
  }
  __syncthreads();
    
   for(int j=0; j<numBins; j++){ 
        
   for (int s = (1+blockDim.x)/2; s>=1; s=s/2){  
       
    if(tid < s){
      collectiveHisto[blockdim*j+tid] += collectiveHisto[blockdim*j+tid+s] ; 

    }  

    __syncthreads();
  }
  histo[j] = collectiveHisto[blockdim*j];
  
  }
}

__global__
void my_histo3_1(int actual_size, int* collectiveHisto, int perThread, int numBlocks, float* logLum, float minLogLum, float logLumRange, const int numBins){


  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  int blockdim = blockDim.x;
  int totalThreads = blockdim*numBlocks;
  
  for(int j= id*perThread; j < (id+1)*perThread; j++){
    if(j<actual_size){
  
      int bin_pre =(int) ((logLum[j] - minLogLum)/logLumRange * numBins);
      int bin = min(bin_pre , numBins-1);
      collectiveHisto[totalThreads*bin + id]+=1;
      //atomicAdd(&(histo[bin]),1); 
  }}
  __syncthreads();
}

__global__
void reduce_sum(const size_t size,  int *array, int *intermed){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int bound = blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  __shared__ int cache[1024];
  cache[tid] = array[id]; 
  __syncthreads();


  for (int s = (blockDim.x +1)/2; s>=1; s=s/2){  
       
    if(tid < s){
      cache[tid] += cache[tid+s];  

    }  

    __syncthreads();
  }
  intermed[blockIdx.x] = cache[0];
}


__global__
void reduce_scan(const int step,  int *array_in, int *array_out){  
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(id > step-1){array_out[id] = array_in[id]+array_in[id-step]; }
  else{array_out[id] = array_in[id];}
}



int main(){
srand (time(NULL));
  //float a[] = {1,1,2,2,4,5,6,7,6,6,6,11,12,6,14,15};
  float *a;
  int init_size=1024*1024-1;
  int poww =0;
  for (int s=init_size-1; s>=1; s=s/2){
    poww = poww+1;
  }
  int postsize = (int) exp2((float)poww); 
  cout << postsize<<"\n";
  int length = postsize;
  a = (float*) malloc(sizeof(float)* length);
  memset(a, 0, sizeof(float)* length);
  random_ints(a,length);
  cout << a[length-1]<<"\n";
  float *d_a;
  int *d_histo;
  int *histo, *d_collectiveHisto;
  int *intermed_b1,*intermed_b2,*intermed_b3,*intermed_b4;
  int numBins = 4;
  int blocks = 8;
  int numthreads = blocks*1024;
  int perThread = length/numthreads;
  
  histo = (int*) malloc(4*sizeof(int));
  memset(histo, 0 , sizeof(int)*4);
  cudaMalloc(&d_histo, sizeof(int)*4);
  cudaMemset(d_histo, 0, sizeof(int)*4);
  cudaMalloc(&d_collectiveHisto, sizeof(int)*4*numthreads);
  cudaMemset(d_collectiveHisto, 0, sizeof(int)*4*numthreads);
  cudaMalloc(&d_a, sizeof(float)*length);
  cudaMemcpy(d_a, a, sizeof(float)*length, cudaMemcpyHostToDevice);
  
  cudaMalloc(&intermed_b1, sizeof(int)*8);
  cudaMalloc(&intermed_b2, sizeof(int)*8);
  cudaMalloc(&intermed_b3, sizeof(int)*8);
  cudaMalloc(&intermed_b4, sizeof(int)*8);
  
  cudaMemset(intermed_b1, 0, sizeof(int)*8);
  cudaMemset(intermed_b2, 0, sizeof(int)*8);
  cudaMemset(intermed_b3, 0, sizeof(int)*8);
  cudaMemset(intermed_b4, 0, sizeof(int)*8);
  
  GpuTimer timer;
  timer.Start();
  
  my_histo3_1<<<8,1024>>>(init_size,d_collectiveHisto,perThread,blocks, d_a, 0, 32.0f, numBins);
  
  reduce_sum<<<8,1024>>>(1024*blocks, d_collectiveHisto, intermed_b1);
  reduce_sum<<<8,1024>>>(1024*blocks, d_collectiveHisto+1024*blocks, intermed_b2);
  reduce_sum<<<8,1024>>>(1024*blocks, d_collectiveHisto+2*1024*blocks, intermed_b3);
  reduce_sum<<<8,1024>>>(1024*blocks, d_collectiveHisto+3*1024*blocks, intermed_b4);
  
  reduce_sum<<<1,8>>>(blocks, intermed_b1, d_histo);
  reduce_sum<<<1,8>>>(blocks, intermed_b2, d_histo+1);
  reduce_sum<<<1,8>>>(blocks, intermed_b3, d_histo+2);
  reduce_sum<<<1,8>>>(blocks, intermed_b4, d_histo+3);
  
  
  //my_histo<<<1024,1024>>>(d_histo,d_a, 0,32, 4 );
  
 //my_histo2<<<1,1024>>>(perThread, d_histo, length, d_a, 0, 32, 4);
  
  timer.Stop();
  float time = timer.Elapsed();
  cout<< time << " \n";
  cudaMemcpy(histo, d_histo, sizeof(int)*4, cudaMemcpyDeviceToHost);
  cout << histo[0] << " "<<histo[1]<< " "<< histo[2]<< " "<< histo[3] << "\n";
  
  
  //int b[] = {1,1,2,2,4,5,6,7,6,6,6,11,12,6,14,15};
  int *b,*d_b, *d_out, *out;
  b = (int*) malloc(sizeof(int)*length);
  out = (int*) malloc(sizeof(int)*length);
  cudaMalloc(&d_b, sizeof(int) * length);
  cudaMalloc(&d_out, sizeof(int)*length);
  cudaMemcpy(d_b,b, sizeof(int)*length, cudaMemcpyHostToDevice);
  
  GpuTimer timer2;
  timer2.Start();
  reduce_scan<<<1024,1024>>>(1,d_b,d_out);
  reduce_scan<<<1024,1024>>>(2,d_out,d_b);
  reduce_scan<<<1024,1024>>>(4,d_b,d_out);
  reduce_scan<<<1024,1024>>>(8,d_out,d_b);
  reduce_scan<<<1024,1024>>>(16,d_b,d_out);
  reduce_scan<<<1024,1024>>>(32,d_out,d_b);
  reduce_scan<<<1024,1024>>>(64,d_b,d_out);
  reduce_scan<<<1024,1024>>>(128,d_out,d_b);
    reduce_scan<<<1024,1024>>>(256,d_b,d_out);
  reduce_scan<<<1024,1024>>>(512,d_out,d_b);
  reduce_scan<<<1024,1024>>>(1024,d_b,d_out);
  reduce_scan<<<1024,1024>>>(2048,d_out,d_b);
  reduce_scan<<<1024,1024>>>(4096,d_b,d_out);
  reduce_scan<<<1024,1024>>>(8192,d_out,d_b);
  reduce_scan<<<1024,1024>>>(16384,d_b,d_out);
  reduce_scan<<<1024,1024>>>(32768,d_out,d_b);
    reduce_scan<<<1024,1024>>>(65536,d_b,d_out);
  reduce_scan<<<1024,1024>>>(65536*2,d_out,d_b);
  reduce_scan<<<1024,1024>>>(65536*4,d_b,d_out);
  reduce_scan<<<1024,1024>>>(65536*8,d_out,d_b);
  
//  out[0]=b[0];
//  for(int i =1;i<length;i++){
//  out[i] = out[i-1]+b[i];
//  } 
  
  timer2.Stop();
  cout<< timer2.Elapsed() << " ms \n";
  out = (int*) malloc(sizeof(int)*length);
  cudaMemcpy(out,d_b, sizeof(int)*length, cudaMemcpyDeviceToHost);
  
//  for(int i=0;i<16; i++){
//  cout<<out[i]<< " \n";}



//////  int length = 512*512;
//////  srand (time(NULL));
//////  float *a;
//////   a=(float *) malloc(length*sizeof(float));
//////   random_ints(a,length);
//////  cout << a[3] << "\n";
//////  //float a[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
//////  float *b, *d_a, *d_b, *d_c, *c;
//////  b = (float*) malloc(512*sizeof(float));
//////  c = (float*) malloc(1*sizeof(float));
//////  
//////  
//////  cudaMalloc(&d_a, length*sizeof(float));
//////  cudaMalloc(&d_b, 512*sizeof(float));
//////  cudaMalloc(&d_c, 1*sizeof(float));
//////  cudaMemcpy(d_a,a,sizeof(float)*length, cudaMemcpyHostToDevice);
//////  GpuTimer timer;
//////  timer.Start();
//////  calculate_max2<<<512,512,512*sizeof(float)>>>(length, d_a,d_b);
////// calculate_max2<<<1,512,512*sizeof(float)>>>(512, d_b,d_c);
////////  for(int i = 0 ; i<length; i++){
////////    if (c[0]<a[i]) {c[0] = a[i];}
////////  
////////  }

//////  timer.Stop();
//////  cout<< timer.Elapsed() << "\n";
//////  cudaMemcpy(c,d_c,sizeof(float), cudaMemcpyDeviceToHost);
//////  cout<<c[0]<< "\n";
  return 0;
}
