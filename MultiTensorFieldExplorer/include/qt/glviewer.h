#ifndef GLVIEWER_H
#define GLVIEWER_H
#include <QGLWidget>
#include <vector>
#include <set>
#include "MultiTensorField.h"
struct RGBTriplet;
class GLViewer:public QGLWidget{
    Q_OBJECT
public:
	float cameraDistance;
	float cameraAngleX;
	float cameraAngleY;
	GLdouble modelMatrix[16];
    unsigned int textureID;
    QPoint lastPos;
	double eyeR;
	double eyeTh;
	double eyePhi;
    int xRot;
    int yRot;
    int zRot;
    int scale;

    double xStartDrag;
    double yStartDrag;
    double zStartDrag;
    double xCenter;
    double yCenter;
    double zCenter;
    double leftBorder;
    double rightBorder;
    double topBorder;
    double bottomBorder;
	bool gearRotationX;
	bool gearRotationY;
    int width;
    int height;
	//-----------shapes---------
	int nShapes;
	int pointsPerShape;
	double *shapes;
	//-------------------------------

	MultiTensorField tensorField;
	std::set<int> visibleSlices;
	std::set<int> visibleTensors;
	void setVisibleSlices(const std::string &s);
	void setVisibleTensors(const std::string &s);


    GLViewer(QWidget *parent = 0);
    ~GLViewer();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    QColor qtPurple;
	QColor qtBlack;
	QColor qtWhite;
    unsigned displayList;


	unsigned char *planeTexture[3];
	GLuint planeTextureID[3];
	bool visiblePlanes[3];
	int currentSlice[3];
	int totalSlices[3];

    void drawMesh(float *nodes, float *normals, int numNodes, int *triangles, int numTriangles, float *nodeColors);
    void clearDisplayList(void);
	//====visualization parameters and functions====
	float	ambient[4];
    float	diffuse[4];
    float	specular[4];
    int		shininess;
	unsigned char **palette;
	int paletteSize;
	void setupLighting(void);
	float position[4];
	void setPalette(RGBTriplet *_palette, int size);
	void setTexture(unsigned char *data, int rows, int cols, int index);
	void setMaskedTexture(unsigned char *data, unsigned char *mask, int rows, int cols, int index);
	void createVoidPlaneTextures(int lx=256, int ly=256, int lz=256);
	void enableTexture(void);
	void setCurrentSlice(int index, int value);
	void setTotalSlices(int index, int value);
	void drawTensor(double *tensor, double px, double py, double pz);
	void loadTensorFieldFromFile(const std::string &fname);
protected:
	
	

    void initializeGL();
	void setDefaultFrustum(void);
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void qNormalizeAngle(int &angle);
    void getWorldCoordinates(int px, int py, double &wx, double &wy, double &wz);
public slots:
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    void adjustScale(int scale);
    void adjustCenter(int dx, int dy);
signals:
	void multiTensorClicked(MultiTensor *tensor, int slice, int row, int col);
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);
    void scaleChanged(int scale);
    void centerChanged(int px, int py);
};
#endif // GLVIEWER_H
