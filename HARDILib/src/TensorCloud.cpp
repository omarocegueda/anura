#include "TensorCloud.h"
#ifdef USE_QT
#include <QtGui>
#include <QGLWidget>
#include <GL/glu.h>
#endif
using namespace std;
TensorCloud::TensorCloud(){
}

TensorCloud::~TensorCloud(){
}

void TensorCloud::build(int numTensort, double *positions, double *alpha, double *dir, int nDir){
}

#ifdef USE_QT

void TensorCloud::draw(void){
	for(unsigned i=0;i<tensors.size();++i){

	}
						
}
#endif
