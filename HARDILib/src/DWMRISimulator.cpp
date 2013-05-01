#include "DWMRISimulator.h"
#include "macros.h"
#include "geometryutils.h"
#include "linearalgebra.h"
#include <math.h>
#include <iostream>
using namespace std;
DWMRISimulator::DWMRISimulator(int _nslices, int _nrows, int _ncols, int _maxCompartments){
	nslices=_nslices;
	nrows=_nrows;
	ncols=_ncols;
	maxCompartments=_maxCompartments;
	int nvox=nslices*nrows*ncols;
	compartmentCount=new int[nvox];
	memset(compartmentCount, 0, sizeof(int)*nvox);
	orientations=new double[nvox*3*maxCompartments];
	distToBoundary=new double[nvox*maxCompartments];
}

DWMRISimulator::~DWMRISimulator(){
	DELETE_ARRAY(compartmentCount);
	DELETE_ARRAY(orientations);
	DELETE_ARRAY(distToBoundary);
}

void DWMRISimulator::addPath(double *path, int npoints, double radius){
	double midSlice=(nslices-1)*0.5;
	double midRow=(nrows-1)*0.5;
	double midCol=(ncols-1)*0.5;
	double minS=-midSlice-0.5;
	double maxS=midSlice+0.5;
	double minR=-midRow-0.5;
	double maxR=midRow+0.5;
	double minC=-midCol-0.5;
	double maxC=midCol+0.5;
	//----validate path----
	double *p=path;
	for(int i=0;i<npoints;++i, p+=3){
		if((p[0]<minC) || (maxC<p[0])){
			cerr<<"Warning: path outside volume (x axis, columns)."<<endl;
		}
		if((p[1]<minR) || (maxR<p[1])){
			cerr<<"Warning: path outside volume (y axis, rows)."<<endl;
		}
		if((p[2]<minS) || (maxS<p[2])){
			cerr<<"Warning: path outside volume (z axis, slices)."<<endl;
		}
	}
	//---------------------
	double voxelPosition[3];
	
	int pos=0;
	double sqrt3=sqrt(3.0);
	for(int s=0;s<nslices;++s){
		voxelPosition[2]=s-midSlice;//z
		for(int r=0;r<nrows;++r){
			voxelPosition[1]=(nrows-1-r)-midRow;//y
			for(int c=0;c<ncols;++c, ++pos){
				voxelPosition[0]=c-midCol;//x
				if(sqrNorm(voxelPosition, 3)>=((nrows*nrows)/4)){
					continue;
				}
				double *A=path;
				double *B=path+3;
				int closestSegment=-1;
				double dist2ToClosest=SQR(radius+1);
				for(int i=1;i<npoints;++i, A+=3, B+=3){
					double dist2=pointToSegmentSQRDistance(A, B, voxelPosition);
					if(dist2<dist2ToClosest){
						dist2ToClosest=dist2;
						closestSegment=i-1;
					}
				}
				if(dist2ToClosest<radius*radius){
					if(compartmentCount[pos]>=maxCompartments){
						cerr<<"Warning: more compartment than expected at voxel ["<<s<<", "<<r<<", "<<c<<"]"<<endl;
					}else{
						double *ori=&orientations[3*pos*maxCompartments+3*compartmentCount[pos]];
						ori[0]=path[3*(closestSegment+1)]-path[3*closestSegment];
						ori[1]=path[3*(closestSegment+1)+1]-path[3*closestSegment+1];
						ori[2]=path[3*(closestSegment+1)+2]-path[3*closestSegment+2];
						normalize<double>(ori,3);
						double dtb=radius-sqrt(dist2ToClosest);
						if(4*dtb*dtb>3){
							dtb=0.5*sqrt3;
						}
						distToBoundary[pos*maxCompartments+compartmentCount[pos]]=dtb;
						++compartmentCount[pos];
					}
				}
			}
		}
	}
}

void DWMRISimulator::buildMultiTensorField(MultiTensorField &M){
	M.buildFromCompartments(nslices, nrows, ncols, maxCompartments, compartmentCount, distToBoundary, orientations);
}
