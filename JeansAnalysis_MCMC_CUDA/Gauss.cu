//======================================================================
//	Gauss.h
//======================================================================
//	define Gauss quadratures (integration)
//======================================================================


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include "functions.h"



__device__ float _Normal(float a, float b, float xx)
{
	return ((b - a) * xx + a + b) / 2.;
}



__device__ float qgaus(float (*func)(float),float a, float b)
{
  int j;
  float xr,xm,dx,s;
  
  static float x[]={0.1488743389816312,0.4333953941292472,
		     0.6794095682990244,0.8650633666889845,0.9739065285171717};
  static float w[]={0.2955242247147529,0.2692667193099963,
		     0.2190863625159821,0.1494513491505806,0.0666713443086881};
    
  xm=0.5*(b+a);
  xr=0.5*(b-a);
  s=0;
  for (j=0;j<5;j++) {
    dx=xr*x[j];
    s += w[j]*((*func)(xm+dx)+(*func)(xm-dx));
  }
  return s *= xr;
}

__device__ float dgl20(float (*func)(float), float a, float b)
{
	static float g[10] = {	0.076526521133497333, 0.227785851141645078,
							0.373706088715419561, 0.510867001950827098,
							0.636053680726515025, 0.746331906460150793,
							0.839116971822218823, 0.912234428251325906,
							0.963971927277913791, 0.993128599185094925};
	static float w[10] = {	0.152753387130725851, 0.149172986472603747,
							0.142096109318382051, 0.131688638449176627,
							0.118194531961518417, 0.101930119817240435,
							0.083276741576704749, 0.062672048334109064,
							0.040601429800386941, 0.017614007139152118};
	float s;
	int i, mflag;

	if(a == b)	return 0.;
	mflag = 0;
	if(a > b)
	{
		mflag = 1;
		s = a;
		a = b;
		b = s;
	}
	for(i = 0, s = 0.; i < 10; i++)	s += w[i] *((*func)(_Normal(a, b, g[i])) + (*func)(_Normal(a, b, -g[i])));
	s *= ((b - a) / 2.);
	if(mflag)	return -s;
	return s;
}
