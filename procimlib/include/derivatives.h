#ifndef DERIVATIVES_H
#define DERIVATIVES_H
enum EDerivativeType{EDT_Forward, EDT_Backward};
enum EBoundaryCondition{EBC_Circular, EBC_DirichletZero, EBC_VonNeumanZero};
int computeGradient(double *f, int nrows, int ncols, double *dfdr, double *dfdc, EDerivativeType edt=EDT_Forward, EBoundaryCondition ebc=EBC_VonNeumanZero);
int computeDivergence(double *fr, double *fc, int nrows, int ncols, double *div, EDerivativeType edt=EDT_Backward, EBoundaryCondition ebc=EBC_VonNeumanZero);

#endif
