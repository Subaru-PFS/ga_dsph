/*
 *
 * Integrand.c
 *
 */

#include <stdio.h>
#include <math.h>
#include "functions.h"

__device__ float X1_func[N],Y1_func[N];
__managed__ float *DataP_d, *DataR_d;//on device(GPU)


__device__ float BetaAni_(float r)
{
//    return (Pnow.bet0_+Pnow.betinf_*powf(r/Pnow.rb_,Pnow.net_)) / (1.+powf(r/Pnow.rb_,Pnow.net_));
    return (DataP_d[8]+DataP_d[9]*powf(r/DataP_d[6],DataP_d[7])) / (1.+powf(r/DataP_d[6],DataP_d[7]));
}

__device__ float IntBetaAni_(float r)
{
//    return powf(r/Pnow.rb_,2.*Pnow.bet0_) * powf(1.+powf(r/Pnow.rb_,Pnow.net_),2*(Pnow.betinf_-Pnow.bet0_)/Pnow.net_);
    return powf(r/DataP_d[6],2.*DataP_d[8]) * powf(1.+powf(r/DataP_d[6],DataP_d[7]),2*(DataP_d[9]-DataP_d[8])/DataP_d[7]);
}


__device__ float StarDist3D_(float r)
{
//    return powf(1.+r*r/(Pnow.rh_*Pnow.rh_),-2.5);
    return powf(1.+r*r/(DataP_d[0]*DataP_d[0]),-2.5);
}

__device__ float DMdens_(float r)
{
//    return Pnow.rhos_ * powf(r/Pnow.rs_,-Pnow.gam_) * powf(1.+powf(r/Pnow.rs_,Pnow.alp_),-(Pnow.bet_-Pnow.gam_)/Pnow.alp_);
    return DataP_d[1] * powf(r/DataP_d[2],-DataP_d[5]) * powf(1.+powf(r/DataP_d[2],DataP_d[3]),-(DataP_d[4]-DataP_d[5])/DataP_d[3]);
}

/*
As reference:
DataP_d[]
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
*/

__device__ float func_(float xi, float chi, float rp)
{
    float f_t;
    float f_, g_;
    float c_up,c_lw;
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    
    c_up = 0.00001;
    c_lw = 0.99999;
    
    f_ = (DataR_d[idx]+xi*xi + chi*chi*(1.-xi*xi))/((1.-chi*chi)*(1.-xi*xi));
    g_ = (DataR_d[idx]+xi*xi)/(1.-xi*xi);
    
    if(xi<=c_up || xi>= c_lw || chi<=c_up || chi>= c_lw){ f_t = 0.0;}
    else
    {
        f_t = xi*(DataR_d[idx]+xi*xi)*powf(1.-xi*xi,-1.) * chi*powf(DataR_d[idx]+xi*xi+chi*chi*(1.-xi*xi),-2.) * powf((DataR_d[idx]+xi*xi)*(DataR_d[idx]+xi*xi)-DataR_d[idx]*DataR_d[idx]*(1.-xi*xi)*(1.-xi*xi),-0.5)*(1.-BetaAni_(g_)*DataR_d[idx]*DataR_d[idx]/g_/g_)*powf(IntBetaAni_(g_),-1.)*IntBetaAni_(f_)*StarDist3D_(f_)*4.*PI*rp*rp*DMdens_(rp);
    }
    return f_t;
    
}


// func integration
__device__ float f3_func_3d(float zz)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    return func_(X1_func[idx], Y1_func[idx], zz);
}

__device__ float f2_func_3d(float yy)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    Y1_func[idx] = yy;
    return qgaus(f3_func_3d,chi_low(X1_func[idx],yy),chi_up(X1_func[idx],yy));//okay up to here
}

__device__ float f1_func_3d(float xx)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    X1_func[idx] = xx;
    return qgaus(f2_func_3d,xi_low(xx),xi_up(xx));
}

__device__ float f0_func_3d()
{
    return dgl20(f1_func_3d,0.,0.9999);
}





__device__ float chi_low(float chi, float xi)
{
    return 0.;
    //return (0.*chi + 0.*xi + 0.);
}

__device__ float chi_up(float chi, float xi)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    //return 1;
    return (DataR_d[idx]+xi*xi + chi*chi*(1.-xi*xi))/((1.-chi*chi)*(1.-xi*xi));
}

__device__ float xi_low(float xi)
{
    return 0.;
    //return (0.*xi + 0.);
}

__device__ float xi_up(float xi)
{
    return 1.;
    //return (0.*xi + 1.);
}
