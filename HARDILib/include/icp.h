#ifndef ICP_H
#define ICP_H
#include "string.h"
//aligns all 3D shapes to reference by a rigid transformation. If reference is NULL, then the first shape is taken as reference
void icp(double *reference, double *shapes, int nshapes, int npoints, double *aligningErrors=NULL);
#endif
