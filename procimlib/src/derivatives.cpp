#include "derivatives.h"
int computeRowDerivative_Forward(double *f, int nrows, int ncols, double *dfdr, EBoundaryCondition ebc){
	for(int j=0;j<ncols;++j){
		for(int i=0;i<nrows-1;++i){
			dfdr[i*ncols+j]=f[(i+1)*ncols+j]-f[i*ncols+j];
		}
	}
	switch(ebc){
		case EBC_Circular:
			for(int j=0;j<ncols;++j){
				dfdr[(nrows-1)*ncols+j]=f[0*ncols+j]-f[(nrows-1)*ncols+j];
			}
		break;
		case EBC_DirichletZero:
			for(int j=0;j<ncols;++j){
				dfdr[(nrows-1)*ncols+j]=0-f[(nrows-1)*ncols+j];
			}
		break;
		case EBC_VonNeumanZero:
			for(int j=0;j<ncols;++j){
				dfdr[(nrows-1)*ncols+j]=0;
			}
		break;
	}
	return 0;
}

int computeColumnDerivative_Forward(double *f, int nrows, int ncols, double *dfdc, EBoundaryCondition ebc){
	for(int i=0;i<nrows;++i){
		for(int j=0;j<ncols-1;++j){
			dfdc[i*ncols+j]=f[i*ncols+j+1]-f[i*ncols+j];
		}
	}
	switch(ebc){
		case EBC_Circular:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols+ncols-1]=f[i*ncols+0]-f[i*ncols+ncols-1];
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols+ncols-1]=0-f[i*ncols+ncols-1];
			}
		break;
		case EBC_VonNeumanZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols+ncols-1]=0;
			}
		break;
	}
	return 0;
}
//------------------------
int computeRowDerivative_Backward(double *f, int nrows, int ncols, double *dfdr, EBoundaryCondition ebc){
	for(int j=0;j<ncols;++j){
		for(int i=1;i<nrows;++i){
			dfdr[i*ncols+j]=f[i*ncols+j]-f[(i-1)*ncols+j];
		}
	}
	switch(ebc){
		case EBC_Circular:
			for(int j=0;j<ncols;++j){
				dfdr[j]=f[j]-f[(nrows-1)*ncols+j];
			}
		break;
		case EBC_DirichletZero:
			for(int j=0;j<ncols;++j){
				dfdr[j]=f[j];
			}
		break;
		case EBC_VonNeumanZero:
			for(int j=0;j<ncols;++j){
				dfdr[j]=0;
			}
		break;
	}
	return 0;
}

int computeColumnDerivative_Backward(double *f, int nrows, int ncols, double *dfdc, EBoundaryCondition ebc){
	for(int i=0;i<nrows;++i){
		for(int j=1;j<ncols;++j){
			dfdc[i*ncols+j]=f[i*ncols+j]-f[i*ncols+j-1];
		}
	}
	switch(ebc){
		case EBC_Circular:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols]=f[i*ncols]-f[i*ncols+ncols-1];
			}
		break;
		case EBC_DirichletZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols]=f[i*ncols];
			}
		break;
		case EBC_VonNeumanZero:
			for(int i=0;i<nrows;++i){
				dfdc[i*ncols]=0;
			}
		break;
	}
	return 0;
}
//---------------------------------

int computeGradient(double *f, int nrows, int ncols, double *dfdr, double *dfdc, EDerivativeType edt, EBoundaryCondition ebc){
	if((nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeRowDerivative_Forward(f, nrows, ncols, dfdr, ebc);
			computeColumnDerivative_Forward(f, nrows, ncols, dfdc, ebc);
		break;
		case EDT_Backward:
			computeRowDerivative_Backward(f, nrows, ncols, dfdr, ebc);
			computeColumnDerivative_Backward(f, nrows, ncols, dfdc, ebc);
		break;
	}
	return 0;
}

int computeDivergence_Forward(double *fr, double *fc, int nrows, int ncols, double *div, EBoundaryCondition ebc){
	for(int i=0;i<nrows-1;++i){
		for(int j=0;j<ncols-1;++j){
			div[i*ncols+j]=(fr[(i+1)*ncols+j]-fr[i*ncols+j])+(fc[i*ncols+j+1]-fc[i*ncols+j]);
		}
	}
	switch(ebc){
		case EBC_Circular:
			div[nrows*ncols-1]=(fr[ncols-1]-fr[nrows*ncols-1])+(fc[(nrows-1)*ncols]-fc[nrows*ncols-1]);
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				div[pos]=(fr[pos+ncols]-fr[pos]) + (fc[pos-(ncols-1)]-fc[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				div[pos]=(fr[j]-fr[pos]) + (fc[pos+1]-fc[pos]);
			}
		break;
		case EBC_DirichletZero:
			div[nrows*ncols-1]=(-fr[nrows*ncols-1])+(-fc[nrows*ncols-1]);
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				div[pos]=(fr[pos+ncols]-fr[pos]) + (-fc[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				div[pos]=(-fr[pos]) + (fc[pos+1]-fc[pos]);
			}
		break;
		case EBC_VonNeumanZero:
			div[nrows*ncols-1]=0;
			for(int i=0, pos=ncols-1;i<nrows-1;++i, pos+=ncols){//last column
				div[pos]=(fr[pos+ncols]-fr[pos]);
			}
			for(int j=0, pos=(nrows-1)*ncols;j<ncols-1;++j, ++pos){//last row
				div[pos]=(fc[pos+1]-fc[pos]);
			}
		break;
	}
	return 0;
}

int computeDivergence_Backward(double *fr, double *fc, int nrows, int ncols, double *div, EBoundaryCondition ebc){
	for(int i=1;i<nrows;++i){
		for(int j=1;j<ncols;++j){
			div[i*ncols+j]=(fr[i*ncols+j]-fr[(i-1)*ncols+j])+(fc[i*ncols+j]-fc[i*ncols+j-1]);
		}
	}
	switch(ebc){
		case EBC_Circular:
			div[0]=(fr[0]-fr[(nrows-1)*ncols+0])+(fc[0]-fc[0+ncols-1]);
			for(int i=1;i<nrows;++i){//first column
				div[i*ncols+0]=(fr[i*ncols]-fr[(i-1)*ncols]) + (fc[i*ncols]-fc[i*ncols + ncols-1]);
			}
			for(int j=1;j<ncols;++j){//first row
				div[j]=(fr[j]-fr[(nrows-1)*ncols+j]) + (fc[j]-fc[j-1]);
			}
		break;
		case EBC_DirichletZero:
			div[0]=fr[0]+fc[0];
			for(int i=1;i<nrows;++i){//first column
				div[i*ncols+0]=(fr[i*ncols]-fr[(i-1)*ncols]) + fc[i*ncols];
			}
			for(int j=1;j<ncols;++j){//first row
				div[j]=fr[j] + (fc[j]-fc[j-1]);
			}
		break;
		case EBC_VonNeumanZero:
			div[0]=0;
			for(int i=1;i<nrows;++i){//first column
				div[i*ncols]=(fr[i*ncols]-fr[(i-1)*ncols]) ;
			}
			for(int j=1;j<ncols;++j){//first row
				div[j]=fc[j]-fc[j-1];
			}
		break;
	}
	return 0;
}


int computeDivergence(double *fr, double *fc, int nrows, int ncols, double *div, EDerivativeType edt, EBoundaryCondition ebc){
	if((nrows<2) || (ncols<2)){
		return -1;
	}
	switch(edt){
		case EDT_Forward:
			computeDivergence_Forward(fr, fc, nrows, ncols, div, ebc);
		break;
		case EDT_Backward:
			computeDivergence_Backward(fr, fc, nrows, ncols, div, ebc);
		break;
	}
	return 0;
}












