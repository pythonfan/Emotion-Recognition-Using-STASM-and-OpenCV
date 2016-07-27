#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include<fstream>
#include<iostream>
#include<windows.h>
#include<atlstr.h>
#include<string>
using namespace std;
using namespace cv;
int main()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
	Mat labelsMat(4, 1, CV_32FC1, labels);

	float trainingData[4][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	//Create 900 x (68 x 2) matrix of training data
	CvMLData cvml;
	
	cvml.read_csv("C:\\Users\\Shakti\\Downloads\\EmotionTrainData\\newSel_anger_ratios_flipped.csv");

	// Indicate the column that has the response
	const CvMat* temp = cvml.get_values();
	int numfeatures = temp->cols - 1;
	cvml.set_response_idx(numfeatures);
	const CvMat* rs = cvml.get_responses();
	//split train data
	CvTrainTestSplit spl((float)0.66);
	cvml.set_train_test_split(&spl);

	// get the respective indices of the training and testing data and store it in the cv::Mat format.
	const CvMat* traindata_idx = cvml.get_train_sample_idx();
	const CvMat* testdata_idx = cvml.get_test_sample_idx();
	Mat mytraindataidx(traindata_idx);
	Mat mytestdataidx(testdata_idx);

	Mat all_Data(temp);
	Mat all_responses = cvml.get_responses();
	Mat traindata(mytraindataidx.cols, numfeatures, CV_32F);
	Mat trainresponse(mytraindataidx.cols, 1, CV_32S);
	Mat testdata(mytestdataidx.cols, numfeatures, CV_32F);
	Mat testresponse(mytestdataidx.cols, 1, CV_32S);

	//Populate train and test data
	for (int i = 0; i<mytraindataidx.cols; i++)
	{
		trainresponse.at<int>(i) = all_responses.at<float>(mytraindataidx.at<int>(i));
		for (int j = 0; j < numfeatures; j++)
		{
			traindata.at<float>(i, j) = all_Data.at<float>(mytraindataidx.at<int>(i), j);
		}
	}
	cout << "Train data initialized";
	for (int i = 0; i<mytestdataidx.cols; i++)
	{
		testresponse.at<int>(i) = all_responses.at<float>(mytestdataidx.at<int>(i));
		for (int j = 0; j < numfeatures; j++)
		{
			testdata.at<float>(i, j) = all_Data.at<float>(mytestdataidx.at<int>(i), j);
		}
	}
	cout << "Test data initialized";

	//Setup SVM parameters
	CvSVM opencv_svm;
	CvSVMParams params;
	params.svm_type = CvSVM::NU_SVC;
	params.C = 0.1;
	params.nu=0.9;
	params.gamma = 3;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);
	//Train
	if (opencv_svm.train_auto(traindata, trainresponse, Mat(), Mat(), params))
	{
		cout << "Train complete";

	}
	else
		cout << "Train complete";
	//save model to file
	opencv_svm.save("C:\\Users\\Downloads\\CK+Dataset\\neutralData\\svm_anger_ratios_flipped.xml");

	//Testing
	int k = 0;
	for (int i = 0; i<testdata.rows; i++)
	{
		Mat samplemat(testdata, Range(i, i + 1));
		float response = opencv_svm.predict(samplemat, false);
		k = (response == testresponse.at<int>(i)) ? ++k : k;
		cout << "Test Response: " << response << " Expected" << testresponse.at<int>(i)<<  "\n";
	}
	cout << "accuracy by the opencv svm is ..." << 100.0 * (float)k / testdata.rows << endl;

}
