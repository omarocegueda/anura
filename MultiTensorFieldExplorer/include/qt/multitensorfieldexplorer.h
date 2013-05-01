#ifndef DTEXPLORER_H
#define DTEXPLORER_H

#include <QtGui/QMainWindow>
#include "ui_multitensorfieldexplorer.h"
class MultiTensorFieldExperimentParameters;
class MultiTensorFieldExplorer : public QMainWindow
{
	Q_OBJECT

public:
	MultiTensorFieldExplorer(QWidget *parent = 0, Qt::WFlags flags = 0);
	~MultiTensorFieldExplorer();
	double *DBFDirections;
	int numDBFDirections;
public slots:
	void updateTensorViewRecovered(void);
	void updateTensorViewGT(void);
	MultiTensorFieldVisualizationType getVisualizationType(void);
	void runDBF(void);
	void loadOnlyGT(void);
	void loadOnlyRecovered(void);
	void saveRecovered(void);
	void selectVoxelFromGT(void);
	void selectVoxelFromRecovered(void);
	void getSelectedVoxelPositionGT(int &slice, int &row, int &col);
	void getSelectedVoxelPositionRecovered(int &slice, int &row, int &col);
	void getUIParameters(MultiTensorFieldExperimentParameters &params);
	void multiTensorSelectedGT(MultiTensor *tensor, int slice, int row, int col);
	void multiTensorSelectedRecovered(MultiTensor *tensor, int slice, int row, int col);
	void testDenoising(void);
	void runSpectralClustering(void);
	void runMultiDimensionalScaling(void);
	void testMask(void);
	void gtFromPaths(void);
	void mtfToLCI(void);
	void LCIToMask(void);
	void createRandomTensor(void);
	void alphaBehaviorStudy(void);
	void evaluate(void);
	void compare(void);
	void buildTrainingSet(void);

private:
	Ui::MultiTensorFieldExplorerClass ui;
};

#endif // DTEXPLORER_H
