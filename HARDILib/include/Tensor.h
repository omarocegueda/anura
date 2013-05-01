#ifndef TENSOR_H
#define TENSOR_H
#include "GDTI.h"
class Tensor{
	protected:
		double position[3];
		double volumeFraction;
		double diffusivities[3];
		double prodDiffusivities;
		double rotationMatrix[9];
		void setDefault(void);
	public:
		Tensor();
		~Tensor();
		//--accessors--
		double getVolumeFraction(void);
		double getDiffusivity(int k);
		double *getDiffusivities(void);
		double *getRotationMatrix(void);
		void setVolumeFraction(double vf);
		void setDiffusivity(int k, double diff);
		void setDiffusivities(double lambda_min, double lambda_middle, double lambda_max);
		void setDiffusivities(double *lambda);
		//------------
		void getPDD(double *pdd)const;
		void setRotationMatrixFromPDD(double *pdd);
		double computeFractionalAnisotropy(void);

		void computeODF(double *directions, int nDirections, double *ODF);
		double computeSignal(double *bCoord);
		void addNoise(double *S, int len, double sigma, double *Sn);
		void acquireWithScheme(double *b, double *gradList, int nDir, double sigma, double *S);
		void acquireWithScheme(double b, double *gradList, int nDir, double sigma, double *S);
		void rotationFromAngles(double azimuth, double zenith);
		
		void fitFromSignal(double S0, double *S, GDTI &H);
		void loadFromTxt(FILE *F);
		void saveToTxt(FILE *F);
};
#endif
