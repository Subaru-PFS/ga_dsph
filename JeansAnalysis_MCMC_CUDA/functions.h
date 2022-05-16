/*
 *
 * function.h
 * Definition of functions and variables
 */

#include <string>
#include <random>

#include<stdio.h>
#include <iostream>
#include <fstream>
#include <string>

//using namespace std;


//number points: up to 1e6
#define N 2560
//Titan RTX: 4608 CUDA cores
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 10 // =N/THREADS_PER_BLOCK

#define PI M_PI   //circular rate
#define GRAVITY_CONST 4.3*pow(10.,-3.)      //gravitational constant(pc*(km/s)^2/M_solar)



//class that holds parameters that are explored in MCMC
class MyParameters {

	float jump(float ParaNow, float jumpscale){
	  std::default_random_engine engine(std::random_device{}());
	  std::normal_distribution<float> distribution(0.0,1.0);
	  
	  if(abs(ParaNow) < 1 ){
	  //if value is zero or too small, we jump a bit
	    return jumpscale * distribution(engine);
	  } else {
	  //otherwise, we jump proportional to value
	    return ParaNow * jumpscale * distribution(engine);
	  }
	}

        float CutMinMax(float ParaNow, float Pmin, float Pmax){
          ParaNow = max(Pmin,ParaNow);//lower limit
          return min(Pmax,ParaNow);//upper limit
	}

public:
	float rh_;  //half light radius !MODIFY IT!
	float R_;   //projected radius from the center of a galaxy
	float rhos_;
	float rs_;
	float alp_;
	float bet_;
	float gam_;
	float rb_;
	float net_;
	float bet0_;
	float betinf_;// free parameters
	float vsys_;
	
	float LL;//log likelihood
	
        
	void RandomNewParameters(float jumpscale){
	  rhos_ += jump(rhos_,jumpscale);
	  rs_ += jump(rs_,jumpscale);
	  alp_ += jump(alp_,jumpscale);
	  bet_ += jump(bet_,jumpscale);
	  gam_ += jump(gam_,jumpscale);
//	  rb_ += jump(rb_,jumpscale); //do not explore velocity asymmetry for now
//	  net_ += jump(net_,jumpscale);
//	  bet0_ += jump(bet0_,jumpscale);
//	  betinf_ += jump(betinf_,jumpscale);
	  vsys_ += jump(vsys_,jumpscale);
	  EnsurePriors();
	}

        void EnsurePriors(){
	  rhos_ = CutMinMax(rhos_, -5.0, 5.0);
          rs_ = CutMinMax(rs_, 1e-2, 1e5);
          alp_ = CutMinMax(alp_, 0.5, 3.0);
          bet_ = CutMinMax(bet_, 2.0, 10.0);
          gam_ = CutMinMax(gam_, -0.5, 1.5);
//        rb_ = CutMinMax(rb_, 1.0, 1e5);//log?
//        net_ = CutMinMax(net_, 1.0, 10.0);
//        bet0_ = CutMinMax(bet0_, 0.0, 1.0);
//        betinf_ = CutMinMax(betinf_, 0.0, 2.0);
          vsys_ = CutMinMax(vsys_, -1000, 1000);
        }

//Get array with values so that it can be handled by CUDA	
        float* GetArray(){
	  static float MyArray[12];
	  MyArray[0] = rh_;
	  MyArray[1] = rhos_;
          MyArray[2] = rs_;
          MyArray[3] = alp_;
          MyArray[4] = bet_;
          MyArray[5] = gam_;
          MyArray[6] = rb_;
          MyArray[7] = net_;
          MyArray[8] = bet0_;
          MyArray[9] = betinf_;
          MyArray[10] = vsys_;
          MyArray[11] = LL;

	  return MyArray;
	}

	void PrintPara(){
	printf("%f %f %f %f %f %f %f %f %f %f %f %f \n",LL,rh_,rhos_,rs_,alp_,bet_,gam_,rb_,net_,bet0_,betinf_,vsys_);
	}

        void WritePara(FILE *myFile){
        fprintf(myFile,"%f %f %f %f %f %f %f %f %f %f %f %f \n",LL,rh_,rhos_,rs_,alp_,bet_,gam_,rb_,net_,bet0_,betinf_,vsys_);
        }

};




// Global variables

//extern MyParameters Pnow;
extern __managed__ float *DataP_d, *DataR_d;//on device(GPU)


// functions
float** read_file(std::string filename,int rows);

void StartChain(int i);
float jump(float ParaNow);

/* Forward Declaration*/
/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

__device__ int myPow(int x, unsigned int p);


__device__ float BetaAni_(float r);
__device__ float IntBetaAni_(float r);
__device__ float StarDist3D_(float r);
__device__ float DMdens_(float r);

// integrand functions
__device__ float func_(float xi, float chi, float rp);
__device__ float f3_func_3d(float zz); // Integrand for 1st integration
__device__ float f2_func_3d(float yy); // Integrand for 2nd integration
__device__ float f1_func_3d(float xx); // Integrand for final integration
__device__ float f0_func_3d(); //Result of triple integration

// Integration ranges
__device__ float chi_low(float chi, float xi);
__device__ float chi_up(float chi, float xi);
__device__ float xi_low(float xi);
__device__ float xi_up(float xi);


//Gauss integration
__device__ float _Normal(float a, float b, float xx);
__device__ float qgaus(float (*func)(float),float a, float b);
__device__ float dgl20(float (*func)(float), float a, float b);
