/*
 *
 * main.c
 *
 */ 

#include <random>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cooperative_groups.h>

#include "functions.h"
using namespace std;

__managed__ float *DataV_d, *DataVerr_d, *LL_d, *LLsum_d;//on device(GPU)
extern __managed__ float *DataP_d, *DataR_d;//also needed for integrals
float LogLikelihood(MyParameters Pnow);


int main(){
    float** DataArray;
    int i,j,istep;
    int Nchains = 5; //number of different chains that shall be explored
    int Nstep = 5000; //number of steps per chain
    bool i_accept;
    float p_accept, MyRand;

    int N_smooth=1000; //number of steps over which acc. rate should be smoothed
    float acc_arr[N_smooth];//running mean of acceptance rate
    float mean_acc;
    float jumpscale;

    FILE *myfile = NULL;
    char buf[0x100];

    float *DataR_h, *DataV_h, *DataVerr_h, *DataP_h, *LL_h, *LLsum_h;//vectors on host(CPU)
    
    // Global variables
//    float rh_;  //half light radius !MODIFY IT!
//    float R_;   //projected radius from the center of a galaxy
//    float rhos_, rs_, alp_, bet_, gam_, rb_, net_, bet0_, betinf_;  // free parameters
//    float parameters[10]; //collect parameters

//    MyParameters Pnow_d;//on GPU (device)
    MyParameters Pnext, Pstart;
    //Starting values: best fit (i.e. how the model was generated)
    MyParameters Pnow;
    Pnow.rh_ = 214.0;//[pc]
    Pnow.rhos_ = 0.5; //[Msun/pc^3]
    Pnow.rs_ = 500.; //[pc]
    Pnow.alp_ = 1.5;
    Pnow.bet_ = 3.0;
    Pnow.gam_ = 0.0;
    Pnow.rb_ = 10000.;
    Pnow.net_ = 5.0;
    Pnow.bet0_ = 0.0;
    Pnow.betinf_ = 0.0;
    Pnow.vsys_ = -290;
    Pstart = Pnow;
    
    
    DataArray = read_file("mockdata/raw_Draco_core_beta_iso_1e4.csv",N);
    //File with 1e4 stars. 1e6 is also available but too big to be hosted on git.
    //Ideally, these files should be created with another script.
    printf("Done reading file \n");
    //CUDA does not like 2D arrays, so we make 3 1D arrays
    DataR_h = (float *) malloc(N * sizeof(float));
    DataV_h = (float *) malloc(N * sizeof(float));
    DataVerr_h = (float *) malloc(N * sizeof(float));
    LL_h = (float *) malloc(N * sizeof(float));
    LLsum_h = (float *) malloc(2*sizeof(float));//at least size 2, otherwise strange

    DataP_h = (float *) malloc(12 * sizeof(float));//holds parameters to be explored by MCMC
    for(i = 0; i < N; i++){
        DataR_h[i] = DataArray[0][i]*1e3;//pc
	DataV_h[i] = DataArray[1][i];
	DataVerr_h[i] = 1.5;//velocity uncertainty
	LL_h[i] = -199.0;
    }
    LLsum_h[0] = -99.0;
    LLsum_h[1] = -199.0;

    DataP_h = Pnow.GetArray();


    //alloc GPU memory
    cudaMalloc(&DataR_d,N*sizeof(float));
    cudaMalloc(&DataV_d,N*sizeof(float));
    cudaMalloc(&DataVerr_d,N*sizeof(float));
    cudaMalloc(&LL_d,N*sizeof(float));
    cudaMalloc(&LLsum_d,2*sizeof(float));
    cudaMalloc(&DataP_d,12*sizeof(float));


    //copy to GPU memory
    cudaMemcpy(DataR_d,DataR_h,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(DataV_d,DataV_h,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(DataVerr_d,DataVerr_h,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(LL_d,LL_h,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(LLsum_d,LLsum_h,2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(DataP_d,DataP_h,12*sizeof(float),cudaMemcpyHostToDevice);

    checkCUDAError("memcpy");


    for(i = 0; i < Nchains; i++){
      snprintf(buf, sizeof(buf), "out_nekoya_MCMC_%d.dat", i);
      myfile = fopen(buf, "w");

      printf("Starting Chain i=%d \n",i);

//initialise running mean of acceptance rate
      for(j = 0; j<N_smooth; j++){
        acc_arr[j] = 0.5;
      }
      jumpscale = 5e-3;//fiducial starting value

      Pnow = Pstart;

      Pnow.LL = LogLikelihood(Pnow)*2.0;//make more negative so that we always accept first jump
      //Pnow.PrintPara();

      //MCMC random walk
      for(istep = 0; istep < Nstep; istep++){
        Pnext = Pnow;
        Pnext.RandomNewParameters(jumpscale);
	//printf("%d \n",istep);
        //Pnext.PrintPara();

        DataP_h = Pnext.GetArray();
        cudaMemcpy(DataP_d,DataP_h,12*sizeof(float),cudaMemcpyHostToDevice);

        Pnext.LL = LogLikelihood(Pnext);

        if(Pnext.LL > Pnow.LL){
//accept jump
                i_accept = true;
		p_accept = 1.0;
        } else {
                p_accept = exp(Pnext.LL - Pnow.LL);

                MyRand = rand() / float(RAND_MAX);
                if(p_accept > MyRand){
                        i_accept = true;
                } else {
                        i_accept = false;
                }
        }
        fprintf(myfile, "%d %d %f ", istep, i_accept, p_accept);
        Pnext.WritePara(myfile);

        if(i_accept){
        //accept new parameters
                Pnow = Pnext;
        }

        acc_arr[istep%N_smooth] = p_accept;//overwrite each N_smooth
//check if jumpscale needs to be adjusted
	if(istep%N_smooth == 0){
          printf("%d \n",istep);
          Pnow.PrintPara();
          mean_acc = 0;
	  for(j = 0; j<N_smooth; j++){
            mean_acc += acc_arr[j];
          }
          mean_acc = mean_acc / N_smooth;
	  printf("Mean acceptance rate over the last %d steps: %f \n",N_smooth,mean_acc);
          if(mean_acc < 0.25) {
	    jumpscale = jumpscale / 1.5 ;
            printf("Decreased jumpscale to: %f \n",jumpscale);
	  }
          if(mean_acc > 0.75) {
	    jumpscale = jumpscale * 1.5 ;
            printf("Increased jumpscale to: %f \n",jumpscale);
	  }
//optimal range around 0.25-0.5?
	}



      }
      fclose(myfile);
    }
    

    cudaFree(DataR_d);
    cudaFree(DataV_d);
    cudaFree(DataVerr_d);
    cudaFree(LL_d);
    cudaFree(LLsum_d);
    cudaFree(DataP_d);

}


// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
__global__ void sumGPU(float *X, float *out) {
//Parallel sumation of 1D array X of size N
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int N_now = N;
    int ii = 1;
    int idx2;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    while(N_now > 1){
      if(idx == 0){
//do these checks only once with one thread
        if(N_now%2 != 0){
//if the current size of the array is not a multiple of 2
//add last element manually
	  N_now = N_now - 1;
	  N_now = N_now/2;
	  X[0] += X[myPow(2,ii)*N_now];
        }

      } else {
	N_now = N_now / 2;
      }
      g.sync();

      if(idx < N_now){
        idx2 = myPow(2,ii)*idx;
        X[idx2] += X[idx2+myPow(2,ii-1)];
      }
      ii += 1;
    }
    g.sync();
    if (idx == 0){
        out[0] = X[0];
    }
}


__device__ int myPow(int x, unsigned int p)
{
  if (p == 0) return 1;
  if (p == 1) return x;
  
  int tmp = myPow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}


__global__ void sumCommSingleBlock(const float *a, float *out) {
    int ii;
    float sum = 0.;
    for (ii = 0; ii < N; ii++) {
       sum += a[ii];
    }
    out[0] = sum;
}


/* Multi-block version of kernel
   calculate LogLikelihood on GPU */
__global__ void GPU_LL(float *DataR_d, float *DataV_d, float *DataVerr_d, float *DataP_d, float *LL_d)
{
    float sigma2;
    float coef;

    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < N)
    {
      coef = (6.*GRAVITY_CONST/DataP_d[0])*(1.+DataR_d[idx]*DataR_d[idx]/DataP_d[0]/DataP_d[0])*(1.+DataR_d[idx]*DataR_d[idx]/DataP_d[0]/DataP_d[0])*(1.+DataR_d[idx])*(1.+DataR_d[idx]);

      sigma2 = coef*f0_func_3d();
      sigma2 += DataVerr_d[idx]*DataVerr_d[idx];//vel uncertainty

      LL_d[idx] = log(2.*PI*sigma2) + (DataV_d[idx] - DataP_d[10])*(DataV_d[idx] - DataP_d[10])/sigma2;
    }
}


float LogLikelihood(MyParameters Pnow){
  /* calculates LogL for how well data x matches to model distirbution */
  float lnLike=0.0;
  float *LLsum;
  LLsum = (float *) malloc(2*sizeof(float));
  LLsum[0] = -99.0;
  LLsum[1] = -199.0;



  /* run the kernel on the GPU */
  dim3 blocksPerGrid(NUM_BLOCKS,1,1);
  dim3 threadsPerBlock(THREADS_PER_BLOCK,1,1);
  GPU_LL<<< blocksPerGrid, threadsPerBlock >>>(DataR_d,DataV_d,DataVerr_d,DataP_d,LL_d);
  cudaDeviceSynchronize();
  //Quick, parallel summation on GPU. Avoid transferring entire array to CPU memory
  sumCommSingleBlock<<<1, 1>>>(LL_d, LLsum_d);

  checkCUDAError("kernel invocation");

  /* copy the result array back to the host */
  cudaMemcpy(LLsum, LLsum_d, 2*sizeof(float), cudaMemcpyDeviceToHost);

  checkCUDAError("memcpy");

  lnLike = -0.5*LLsum[0];
  return lnLike;
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
