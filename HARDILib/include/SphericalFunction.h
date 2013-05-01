#ifndef SPHERICALFUNCTION_H
#define SPHERICALFUNCTION_H
class SphericalFunction{
	public:
		double evaluate(double *direction)=0;
		double evaluatePrecomputed(int n)=0;
		void precompute(double *directions, int n)=0;
		SphericalFunction();
		~SphericalFunction();
};
#endif
