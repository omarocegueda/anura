#ifndef TENSORCLOUD_H
#define TENSORCLOUD_H
#include "Tensor.h"
#include <vector>
class TensorCloud{
	private:
		std::vector<Tensor> tensors;
	public:
		TensorCloud();
		~TensorCloud();
		void build(int numTensort, double *positions, double *alpha, double *dir, int nDir);
#ifdef USE_QT
		void draw(void);
#endif
};

#endif
