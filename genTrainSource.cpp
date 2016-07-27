// minimal.cpp: Display the landmarks of a face in an image.
//              This demonstrates stasm_search_single.

#include <stdio.h>
#include <stdlib.h>
#include "opencv/highgui.h"
#include "stasm_lib.h"
#include<iostream>
#include<fstream>
#include<atlstr.h>
#include<windows.h>
#include<math.h>
using namespace std;

int main()
{
	
	CString basedirpath = "C:\\Users\\Shakti\\Documents\\myTrainImages\\neutral_anger";
	HANDLE dir;
	WIN32_FIND_DATA file_data;
	dir = FindFirstFile(basedirpath + "\\*", &file_data);
	if (INVALID_HANDLE_VALUE == dir)
	{
		cout << "Handle error";
		return 0;
	}

	//static const char* path = "C:\\Users\\Shakti\\Downloads\\CK+Dataset\\extended-cohn-kanade-images\\cohn-kanade-images\\S010\\002\\S010_002_00000001.png";
	ofstream trainlandmarks;
	trainlandmarks.open("C:\\Users\\Shakti\\Downloads\\EmotionTrainData\\newtrain_anger_neutral.csv");

	
	do
	{
		
		//char basepath[100] = { '\0' };
		//strcat_s(basepath, 64, "C:\\Users\\Shakti\\Documents\\myTrainImages\\happy\\");
		string basepath = "C:\\Users\\Shakti\\Documents\\myTrainImages\\neutral_anger\\";
		string filename(file_data.cFileName);
		basepath.append(filename);
		const char* path = basepath.c_str();
		
		if (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) //if file is a directory
		{
		}
		else
		{
			cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));

			if (!img.data)
			{
				printf("Cannot load %s\n", path);
				exit(1);
			}

			int foundface;
			float landmarks[2 * stasm_NLANDMARKS]; // x,y coords

			if (!stasm_search_single(&foundface, landmarks,
				(char*)img.data, img.cols, img.rows, path, "C:\\Users\\Shakti\\Downloads\\stasm4.1.0\\data"))
			{
				printf("Error in stasm_search_single: %s\n", stasm_lasterr());
				exit(1);
			}

			if (!foundface)
				printf("No face found in %s\n", path);
			else
			{
				// draw the landmarks on the image as white dots
				stasm_force_points_into_image(landmarks, img.cols, img.rows);
				for (int i = 0; i < stasm_NLANDMARKS; i++)
				{
					img(cvRound(landmarks[i * 2 + 1]), cvRound(landmarks[i * 2])) = 255;
						cout << (landmarks[i*2] - landmarks[0])<<" "; 
						float xscale = abs(landmarks[47]-landmarks[15]), yscale = abs(landmarks[22]-landmarks[0]);
						float tmpvalx = (landmarks[i * 2 + 1] - landmarks[47]), tmpvaly = (landmarks[i * 2] - landmarks[0]);
						if (!trainlandmarks.is_open())
							cout << "Outfile not open";
						trainlandmarks << (tmpvaly);
						trainlandmarks << ",";
						cout << tmpvaly<<" ";
						trainlandmarks << (tmpvalx);
						trainlandmarks << ",";
				}
				trainlandmarks << "0\n";
				cout << "\n";
			}
			cv::imshow("stasm minimal", img);
		}//end else
	} while (FindNextFile(dir, &file_data) != 0);
	trainlandmarks.close();
	cv::waitKey();
	return 0;
}
