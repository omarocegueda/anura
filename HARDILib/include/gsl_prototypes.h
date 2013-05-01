#ifndef GSL_PROTOTYPES_H
#define GSL_PROTOTYPES_H
#ifdef USE_GSL
	#include "gsl\gsl_sf_legendre.h"
#else
	#include <iostream>
	double  gsl_sf_legendre_sphPlm(const int l, const int m, const double x){std::cerr<<"Attempting to use unsuported GSL function."<<std::endl;return 0;}
#endif
#endif
