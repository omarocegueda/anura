#include <QtGui>
#include <QGLWidget>
#include "glviewer.h"
#include "utilities.h"
#include "rgbtriplet.h"
#include "linearalgebra.h"
#include "geometryutils.h"
#include <sstream>
#include <GL/glu.h>
#define ZNEAR 0.1
#define ZFAR  1000
using namespace std;
#define USE_OLD_CONTROLS

void setCamera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ)
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(posX, posY, posZ, targetX, targetY, targetZ, 0, 1, 0); // eye(x,y,z), focal(x,y,z), up(x,y,z)
}

GLViewer::GLViewer(   QWidget *parent) :
                            QGLWidget(QGLFormat(QGL::SampleBuffers), parent){
    xRot=0;
    yRot=0;
    zRot=0;
	eyeTh=0;
	eyePhi=0;
	eyeR=1.0;
    scale=-500;
    xStartDrag=0;
    yStartDrag=0;
    zStartDrag=0;
    xCenter=0;
    yCenter=0;
    zCenter=0;
    qtPurple = QColor::fromCmykF(0.39, 0.39, 0.0, 0.0);
	qtBlack = QColor::fromCmykF(0.0, 0.0, 0.0, 0.0);
	qtWhite = QColor::fromCmykF(1.0, 1.0, 1.0, 0.0);
	position[0]=5;
	position[1]=5;
	position[2]=-0.5;
	position[3]=0;
	for(int i=0;i<3;++i){
		ambient[i]=0.3f;
		diffuse[i]=0.9f;
		specular[i]=1000.0f;
	}
	ambient[3]=1;
	diffuse[3]=1;
	specular[3]=1;
    shininess=32;
	for(int i=0;i<3;++i){
		planeTexture[i]=NULL;
		visiblePlanes[i]=false;
		planeTextureID[i]=0;
		currentSlice[i]=128;
		totalSlices[i]=256;
	}
	gearRotationX=false;
	gearRotationY=false;
	memset(modelMatrix, 0, sizeof(modelMatrix));
	modelMatrix[0]=modelMatrix[5]=modelMatrix[10]=modelMatrix[15]=exp(-500.0*0.0025);
	palette=NULL;
	paletteSize=0;

	cameraDistance=0;
	cameraAngleX=0;
	cameraAngleY=0;
	setCamera(0, 0, 10, 0, 0, 0);

	shapes=NULL;
	nShapes=0;
	pointsPerShape=0;
}

GLViewer::~GLViewer(){
	for(int i=0;i<3;++i){
		freeArray(planeTexture[i]);
	}
}

void GLViewer::initializeGL(){
	createVoidPlaneTextures(256, 256, 256);
	qglClearColor(qtWhite);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);
    glShadeModel(GL_SMOOTH);
	setupLighting();
    glEnable(GL_NORMALIZE);
    glLoadIdentity();
    //glMatrixMode(GL_PROJECTION);
	//glClearColor (0.3f,0.3f,0.7f,1.0f);
	//glClearColor (0.0f,0.0f,0.0f,1.0f);
	glClearColor (1.0f,1.0f,1.0f,1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	enableTexture();
}

void GLViewer::setDefaultFrustum(void){
	const float ar = double(this->width) / double(this->height);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(45.0, ar, .1, 1000.0);
   glMatrixMode(GL_MODELVIEW);
}

void GLViewer::paintGL(void){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
#ifdef USE_OLD_CONTROLS
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    glTranslated(xCenter, yCenter, zCenter);
    double s=exp(scale*0.0025);
	//glTranslated(0, 0, -s);
    glScalef(s,s,s);
    glRotated(xRot / 16.0, 1.0, 0.0, 0.0);
    glRotated(yRot / 16.0, 0.0, 1.0, 0.0);
    glRotated(zRot / 16.0, 0.0, 0.0, 1.0);
#else
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double s=0.1;
	glScalef(s,s,s);
	glTranslatef(0, 0, cameraDistance);
    glRotatef(cameraAngleX, 1, 0, 0);   // pitch
    glRotatef(cameraAngleY, 0, 1, 0);   // heading
	glMatrixMode(GL_MODELVIEW);
#endif
	if(visiblePlanes[0]){
		float geometricPosition=2*double(currentSlice[0])/double(totalSlices[0])-1;
		glBindTexture( GL_TEXTURE_2D, planeTextureID[0]);
		glBegin( GL_QUADS );
			glTexCoord2d(0.0,0.0); glVertex3d( geometricPosition, -1.0, -1.0);
			glTexCoord2d(1.0,0.0); glVertex3d( geometricPosition,  1.0, -1.0);
			glTexCoord2d(1.0,1.0); glVertex3d( geometricPosition,  1.0,  1.0);
			glTexCoord2d(0.0,1.0); glVertex3d( geometricPosition, -1.0,  1.0);
		glEnd();
		glBegin( GL_QUADS );
			glTexCoord2d(0.0,0.0); glVertex3d( geometricPosition, -1.0, -1.0);
			glTexCoord2d(0.0,1.0); glVertex3d( geometricPosition, -1.0,  1.0);
			glTexCoord2d(1.0,1.0); glVertex3d( geometricPosition,  1.0,  1.0);
			glTexCoord2d(1.0,0.0); glVertex3d( geometricPosition,  1.0, -1.0);
		glEnd();
	}
	
	if(visiblePlanes[1]){
		float geometricPosition=2*double(currentSlice[1])/double(totalSlices[1])-1;
		glBindTexture( GL_TEXTURE_2D, planeTextureID[1]);
		glBegin( GL_QUADS );
			glTexCoord2d(0.0,0.0); glVertex3d(-1.0,  geometricPosition, -1.0);
			glTexCoord2d(1.0,0.0); glVertex3d( 1.0,  geometricPosition, -1.0);
			glTexCoord2d(1.0,1.0); glVertex3d( 1.0,  geometricPosition,  1.0);
			glTexCoord2d(0.0,1.0); glVertex3d(-1.0,  geometricPosition,  1.0);
		glEnd();
		glBegin( GL_QUADS );
			glTexCoord2d(0.0,0.0); glVertex3d(-1.0,  geometricPosition, -1.0);
			glTexCoord2d(0.0,1.0); glVertex3d(-1.0,  geometricPosition,  1.0);
			glTexCoord2d(1.0,1.0); glVertex3d( 1.0,  geometricPosition,  1.0);
			glTexCoord2d(1.0,0.0); glVertex3d( 1.0,  geometricPosition, -1.0);
		glEnd();
	}
	
	if(visiblePlanes[2]){
		float geometricPosition=2*double(currentSlice[2])/double(totalSlices[2])-1;
		glBindTexture( GL_TEXTURE_2D, planeTextureID[2]);
		glBegin( GL_QUADS );
			glTexCoord2d(0.0,0.0); glVertex3d(-1.0, -1.0, geometricPosition);
			glTexCoord2d(1.0,0.0); glVertex3d( 1.0, -1.0, geometricPosition);
			glTexCoord2d(1.0,1.0); glVertex3d( 1.0,  1.0, geometricPosition);
			glTexCoord2d(0.0,1.0); glVertex3d(-1.0,  1.0, geometricPosition);
		glEnd();
		glBegin( GL_QUADS );
			glTexCoord2d(0.0,0.0); glVertex3d(-1.0, -1.0, geometricPosition);
			glTexCoord2d(0.0,1.0); glVertex3d(-1.0,  1.0, geometricPosition);
			glTexCoord2d(1.0,1.0); glVertex3d( 1.0,  1.0, geometricPosition);
			glTexCoord2d(1.0,0.0); glVertex3d( 1.0, -1.0, geometricPosition);
		glEnd();
	}
	//tensorField.drawTensorField();
	if(shapes==NULL){
		tensorField.normalizeLatticeSeparations();
		for(set<int>::iterator s=visibleSlices.begin();s!=visibleSlices.end();++s){
			tensorField.drawLatticeSlice(*s);
			for(set<int>::iterator k=visibleTensors.begin();k!=visibleTensors.end();++k){
				tensorField.drawPDDSlice(*s,*k);
			}
			
		}
		glPopMatrix();
	}else{
		int resolution=10;
		GLUquadricObj* obj = gluNewQuadric();
		glMatrixMode(GL_MODELVIEW);
		for(int i=0;i<nShapes;++i){
			double *shape=&shapes[i*3*pointsPerShape];
			for(int j=0;j<pointsPerShape;++j){
				double *coords=&shape[3*j];
				double T[16]={
					1, 0	, 0	, 0,
					0, 1	, 0	, 0,
					0, 0	, 1	, 0,
					coords[0]	,	coords[1]	,	coords[2]	, 1
				};
				glPushMatrix();
				glLoadMatrixd(T);
				//glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pddColor);
				gluSphere(obj, 0.0075, resolution, resolution);
				glPopMatrix();
			}
		}
		gluDeleteQuadric(obj);
	}
 }

void GLViewer::resizeGL(int width, int height){
	glMatrixMode(GL_PROJECTION);
    rightBorder = (double)width/(double)height;
    leftBorder  = -rightBorder;
    topBorder   = 1;
    bottomBorder= -topBorder;
    glViewport(0,0,width, height);
    this->width=width;
    this->height=height;
	double ar=double(width)/double(height);
	glLoadIdentity();
	gluPerspective(45.0, ar, ZNEAR, ZFAR);
    glMatrixMode(GL_MODELVIEW);
}
#define SQR(x) ((x)*(x))
double sqrDistance(double x0, double y0, double z0, double x1, double y1, double z1){
    return SQR(x0-x1)+SQR(y0-y1)+SQR(z0-z1);
}

void unProject(int winx, int winy, double &px, double &py, double &pz){
    double model[16];
    double proj[16];
    int view[16];

    glGetDoublev(GL_MODELVIEW_MATRIX,model);
    glGetDoublev(GL_PROJECTION_MATRIX,proj);
    glGetIntegerv(GL_VIEWPORT,view);

    gluUnProject(winx, winy, 0,model,proj,view,&px, &py, &pz);
}

void unProject(int winx, int winy, int winz, double *P){
    double model[16];
    double proj[16];
    int view[16];

    glGetDoublev(GL_MODELVIEW_MATRIX,model);
    glGetDoublev(GL_PROJECTION_MATRIX,proj);
    glGetIntegerv(GL_VIEWPORT,view);

	double px,py,pz;
    gluUnProject(winx, winy, winz,model,proj,view,&px, &py, &pz);
	P[0]=px;
	P[1]=py;
	P[2]=pz;
}

void GLViewer::mousePressEvent(QMouseEvent *event){
    lastPos = event->pos();
    if ((event->buttons() & Qt::RightButton) && !(event->buttons() & Qt::LeftButton)){
        getWorldCoordinates(event->x(), event->y(), xStartDrag, yStartDrag, zStartDrag);
    }
	GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
	int w=viewport[2];
	int h=viewport[3];
	int cx=w/2 + viewport[0];
	int cy=h/2 + viewport[1];
	
	if(abs(event->x()-cx)*4 > w){
		gearRotationY=true;
	}
	if(abs(event->y()-cy)*4 > h){
		gearRotationX=true;
	}
	if(event->buttons() & Qt::MidButton){
		//look for nearest node
		double A[3];
		double B[3];
        unProject(event->x(), this->height-1-event->y(),-1, A);
		unProject(event->x(), this->height-1-event->y(), 1, B);

        
		int nslices=tensorField.getNumSlices();
		int nrows=tensorField.getNumRows();
		int ncols=tensorField.getNumCols();
		int nvoxels=nslices*nrows*ncols;
		int cs=nslices/2;
		int cc=ncols/2;
		int cr=nrows/2;
		int pos=0;
		MultiTensor *voxels=tensorField.getVoxels();
		double best=-1;
        int bestNode=-1;
		int bestSlice=-1;
		int bestRow=-1;
		int bestCol=-1;

		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++pos){
					voxels[pos].setSelected(false);
					double location[3]={j-cc,nrows-1-i-cr,s-cs};
					double opc=pointToLineDistance<double>(location, A, B);
					if((best<0) || (opc<best)){
						bestNode=pos;
						best=opc;
						bestSlice=s;
						bestRow=i;
						bestCol=j;
					}
				}
			}
		}
		voxels[bestNode].setSelected(true);
		emit multiTensorClicked(&voxels[bestNode], bestSlice, bestRow, bestCol);
    }
	updateGL();
}

void GLViewer::mouseReleaseEvent(QMouseEvent *event){
    xStartDrag=0;
    yStartDrag=0;
    zStartDrag=0;
	if((event->buttons() & Qt::RightButton) || (event->buttons() & Qt::LeftButton) ){ 
		emit mousePressEvent(event);
	}
}

void GLViewer::mouseMoveEvent(QMouseEvent *event){
	if((event->x()>=this->width) || (event->y()>=this->height)){
		return;
	}
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();
#ifdef USE_OLD_CONTROLS
	dx*=-1;
	dy*=-1;
#endif
    if((dx==0) && (dy==0)){
        return;
    }
    if((event->buttons() & Qt::LeftButton) && (event->buttons() & Qt::RightButton)){
        adjustScale(-dy);
		cameraDistance += dy * 0.2f;
    }else if (event->buttons() & Qt::LeftButton) {
        setXRotation(xRot + 8 * dy);
        setYRotation(yRot + 8 * dx);
		cameraAngleY += dx;
        cameraAngleX += dy;
    }else if (event->buttons() & Qt::RightButton) {
        adjustCenter(event->x(), event->y());
    }
    lastPos = event->pos();
}


void GLViewer::qNormalizeAngle(int &angle){
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void GLViewer::setXRotation(int angle){

	//-----------
#ifdef USE_OLD_CONTROLS
	glPushMatrix();
	glLoadMatrixd(modelMatrix);
	glRotated((angle-xRot)/16.0, 1.0, 0.0, 0.0); 
	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glPopMatrix();
#endif
	//------------

    qNormalizeAngle(angle);
    if(angle != xRot){
        xRot = angle;
		eyeTh=angle*M_PI/(360.0*16);
        //emit xRotationChanged(angle);
        updateGL();
    }
}

void GLViewer::setYRotation(int angle){
	//-----------
#ifdef USE_OLD_CONTROLS
	glPushMatrix();
	glLoadMatrixd(modelMatrix);
	glRotated((angle-yRot)/16.0, 0.0, 1.0, 0.0); 
	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glPopMatrix();
#endif
	//------------
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
		eyePhi=angle*M_PI/(360.0*16);
        //emit yRotationChanged(angle);
        updateGL();
    }
}

void GLViewer::setZRotation(int angle){
	//-----------
	glPushMatrix();
	glLoadMatrixd(modelMatrix);
	glRotated((angle-zRot)/16.0, 0.0, 0.0, 1.0); 
	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glPopMatrix();
	//------------

    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        emit zRotationChanged(angle);
        updateGL();
    }
}

void GLViewer::adjustScale(int s){
	eyeR+=s;
#ifdef USE_OLD_CONTROLS
	/*glPushMatrix();
	glLoadMatrixd(modelMatrix);
	double ss=exp(s*0.0025);
	glScaled(ss,ss,ss);
	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glPopMatrix();*/
#endif
    scale+=s;
    //emit scaleChanged(s);
    updateGL();
}

void GLViewer::getWorldCoordinates(int px, int py, double &wx, double &wy, double &wz){
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    wx=(double)(px-viewport[0])/(double)(viewport[2]);
    wy=(double)(py-viewport[1])/(double)(viewport[3]);
    wx=leftBorder + wx*(rightBorder-leftBorder);
    wy=topBorder  + wy*(bottomBorder-topBorder);
    wz = ZNEAR;
}


void GLViewer::adjustCenter(int px, int py){
#ifdef USE_OLD_CONTROLS
    double xw, yw,zw;
    getWorldCoordinates(px, py, xw, yw, zw);
    xCenter+=(xw-xStartDrag);
    yCenter+=(yw-yStartDrag);
	
	//-----------
	glPushMatrix();
	glLoadMatrixd(modelMatrix);
	glTranslated(xw-xStartDrag,yw-yStartDrag,0); 
	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glPopMatrix();
	//------------
    xStartDrag=xw;
    yStartDrag=yw;
    zStartDrag=zw;
    emit centerChanged(px, py);
    updateGL();
#endif
}


QSize GLViewer::minimumSizeHint() const
 {
     return QSize(50, 50);
 }

 QSize GLViewer::sizeHint() const
 {
     return QSize(400, 400);
 }

 void GLViewer::drawMesh(float *nodes, float *normals, int numNodes, int *triangles, int numTriangles, float *nodeColors){
	float averageColor[3]={0.5, 0.5, 0.5};
     glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	 glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, averageColor);
	 
     float *nodeColor[3];
     for(int i=0;i<numTriangles;++i){
         if(nodeColors!=NULL){
             nodeColor[0]= &nodeColors[3*triangles[3*i]];
             nodeColor[1]= &nodeColors[3*triangles[3*i+1]];
             nodeColor[2]= &nodeColors[3*triangles[3*i+2]];
             for(int j=0;j<3;++j){
                 averageColor[j]=0;
                 for(int k=0;k<3;++k){
                     averageColor[j]+=nodeColor[k][j];
                 }
                 averageColor[j]/=3.0;
             }
             glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, averageColor);
         }

         float mcolorWhite[] = { 1.0f, 1.0f, 1.0f, 1.0f };
         glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,mcolorWhite);

		 //glBindTexture( GL_TEXTURE_2D, currentCoronalSlice );

         glBegin(GL_TRIANGLE_FAN);
                 glNormal3fv(&normals[i*3]);
                 glVertex3fv(&nodes[triangles[3*i]*3]);
                 glVertex3fv(&nodes[triangles[3*i+1]*3]);
                 glVertex3fv(&nodes[triangles[3*i+2]*3]);
         glEnd();
     }
 }


 void GLViewer::clearDisplayList(void){
     glDeleteLists(displayList,1);
     displayList=0;
 }



void GLViewer::setupLighting(void){
	glEnable(GL_LIGHTING);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT,ambient);
    glLightfv( GL_LIGHT0, GL_AMBIENT, ambient );
    glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuse );
    glLightfv( GL_LIGHT0, GL_SPECULAR, specular );
    glMateriali(GL_FRONT_AND_BACK,GL_SHININESS,shininess);
    glEnable(GL_LIGHT0);
}

void GLViewer::createVoidPlaneTextures(int lx, int ly, int lz){
	int lenPlane[3];
	int rows[3]={lz, lz, ly};
	int cols[3]={ly, lx, lx};
	unsigned char colors[3][3]={{255, 0, 0},{0, 255, 0},{0, 0, 255}};
	for(int i=0;i<3;++i){
		lenPlane[i]=rows[i]*cols[i];
		freeArray(planeTexture[i]);
		newArray<unsigned char>(planeTexture[i], 4*lenPlane[i]);
	}
	for(int k=0;k<3;++k){
		for(int i=0;i<lenPlane[k];++i){
			for(int c=0;c<3;++c){
				planeTexture[k][4*i+c]=colors[k][c];
			}
			planeTexture[k][4*i+3]=0;
		}
		setTexture(planeTexture[k], rows[k], cols[k], k);
	}
}


void GLViewer::enableTexture(void){
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	// when texture area is small, bilinear filter the closest mipmap
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST );
	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	//glEnable( GL_TEXTURE_2D );
}



void GLViewer::setTexture(unsigned char *data, int rows, int cols, int index){
	if((index<0) || (index>=3)){
		return;
	}
	if(planeTexture[index]==NULL){
		newArray<unsigned char>(planeTexture[index], 4*rows*cols);
	}
	memcpy(planeTexture[index], data, 4*rows*cols*sizeof(unsigned char));
	if(planeTextureID[index]==0){
		GLuint textureID=0;
		glGenTextures(1, &textureID);
		planeTextureID[index]=textureID;
	}
	glBindTexture(GL_TEXTURE_2D, planeTextureID[index]);
#if defined(_WIN32) || defined(_WIN64)
	gluBuild2DMipmaps( GL_TEXTURE_2D, 3, cols, rows, GL_RGBA, GL_UNSIGNED_BYTE,  planeTexture[index]);
#endif
}

void GLViewer::setMaskedTexture(unsigned char *data, unsigned char *mask, int rows, int cols, int index){
	if((index<0) || (index>=3)){
		return;
	}
	if(planeTexture[index]==NULL){
		newArray(planeTexture[index], 4*rows*cols*sizeof(unsigned char));
	}
	memcpy(planeTexture[index], data, 4*rows*cols*sizeof(unsigned char));
	for(int i=0;i<rows;++i){
		for(int j=0;j<cols;++j){
			if(mask[i*cols+j]>0){
				unsigned char currentColor=mask[i*cols+j];
				unsigned char *destination=&(planeTexture[index][4*(i*cols+j)]);
				memcpy(destination, palette[currentColor], sizeof(unsigned char)*3);
				destination[3]=0;
			}
		}
	}
	if(planeTextureID[index]==0){
		GLuint textureID=0;
		glGenTextures(1, &textureID);
		planeTextureID[index]=textureID;
	}
	glBindTexture(GL_TEXTURE_2D, planeTextureID[index]);
	gluBuild2DMipmaps( GL_TEXTURE_2D, 3, cols, rows, GL_RGBA, GL_UNSIGNED_BYTE,  planeTexture[index]);
}



void GLViewer::setCurrentSlice(int index, int value){
	if((index<0) || (index>=3)){
		return;
	}
	currentSlice[index]=value;
}

void GLViewer::setTotalSlices(int index, int value){
	if((index<0) || (index>=3)){
		return;
	}
	totalSlices[index]=value;
}

void GLViewer::setPalette(RGBTriplet *_palette, int size){
	if((palette!=NULL) && (paletteSize!=size)){
		freeMatrix<unsigned char>(palette, paletteSize, 3);
	}
	if(palette==NULL){
		newMatrix<unsigned char>(palette, size, 3);
		paletteSize=size;
	}

	for(int i=0;i<paletteSize;++i){
		palette[i][0]=_palette[i].r;
		palette[i][1]=_palette[i].g;
		palette[i][2]=_palette[i].b;
	}
}

void GLViewer::drawTensor(double *T, double px, double py, double pz){
	double M[9]={
		T[0], T[3], T[4],
		T[3], T[1], T[5],
		T[4], T[5], T[2],
	};
	double eigenvalues[3];
	symmetricEigenDecomposition(M, eigenvalues, 3);
	double maxVal=-1;
	for(int i=0;i<3;++i){
		eigenvalues[i]=MAX(eigenvalues[i], 0);
		maxVal=MAX(eigenvalues[i], maxVal);
	}
	for(int i=0;i<3;++i){
		eigenvalues[i]=0.1*eigenvalues[i]/maxVal;
	}
	
	double V[16]={
		M[0], M[1]	, M[2]	, 0,
		M[3], M[4]	, M[5]	, 0,
		M[6], M[7]	, M[8]	, 0,
		px	,	py	,	pz	, 1
	};
	//glPushMatrix();
		//glLoadIdentity();
		//glTranslated(px, py, pz);
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadMatrixd(V);
			//glTranslated(px, py, pz);
			GLUquadricObj* obj = gluNewQuadric();
			glScaled(eigenvalues[0], eigenvalues[1], eigenvalues[2]);
			gluSphere(obj, 1.0, 10, 10);
			gluDeleteQuadric(obj);
			glPopMatrix();
			glMatrixMode(GL_PROJECTION);
	//glPopMatrix();

}

void GLViewer::loadTensorFieldFromFile(const std::string &fname){
	tensorField.loadFromTxt(fname);
}


void GLViewer::setVisibleSlices(const std::string &s){
	visibleSlices.clear();
	istringstream is(s);
	int next;
	while(is>>next){
		visibleSlices.insert(next);
	}
}

void GLViewer::setVisibleTensors(const std::string &s){
	visibleTensors.clear();
	istringstream is(s);
	int next;
	while(is>>next){
		visibleTensors.insert(next);
	}
}
