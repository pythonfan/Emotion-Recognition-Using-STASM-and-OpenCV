#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<algorithm>
using namespace std;

void main()
{
	ifstream inlandmarks;
	ofstream selFeatures;
	inlandmarks.open("C:\\Users\\Shakti\\Downloads\\EmotionTrainData\\newtrain_anger.csv");
	selFeatures.open("C:\\Users\\Shakti\\Downloads\\EmotionTrainData\\newSel_anger_ratios_flipped.csv");
	if (inlandmarks.is_open())
	{
		string line;
		while (getline(inlandmarks, line))
		{
			//cout << line << '\n';
			float datavalues[154];
			int i = 0;
			stringstream ssin(line);
			string tmp;
			while (getline(ssin, tmp, ',' )){
				datavalues[i] = atof(tmp.c_str());
				cout << datavalues[i] << " " ;
				++i;
			}
			cout << "\n";
			//Write features into selFeatures file
			float xscale = max(abs(datavalues[25] - datavalues[1]), abs(datavalues[24]-datavalues[0]));
			float yscale = max((datavalues[12] - datavalues[28]) , abs(datavalues[13] - datavalues[29]));

			selFeatures << ((datavalues[118] - datavalues[130])/yscale) << "," << ((datavalues[119] - datavalues[131])/xscale) << ",";
			selFeatures << ((datavalues[124] - datavalues[148]) / yscale) << "," << ((datavalues[125] - datavalues[149]) / xscale) << ",";
			selFeatures << ((datavalues[112] - datavalues[124]) / yscale) << "," << ((datavalues[113] - datavalues[125]) / xscale) << ",";
			selFeatures << ((datavalues[112] - datavalues[148]) / yscale) << "," << ((datavalues[113] - datavalues[149]) / xscale) << ",";
			selFeatures << ((datavalues[118] - datavalues[116]) / yscale) << "," << ((datavalues[119] - datavalues[117]) / xscale) << ",";
			selFeatures << ((datavalues[130] - datavalues[110]) / yscale) << "," << ((datavalues[131] - datavalues[111]) / xscale) << ",";
			selFeatures << ((datavalues[60] - datavalues[118]) / yscale) << "," << ((datavalues[61] - datavalues[119]) / xscale) << ",";
			selFeatures << ((datavalues[80] - datavalues[130]) / yscale) << "," << ((datavalues[81] - datavalues[131]) / xscale) << ",";
			selFeatures << ((datavalues[68] - datavalues[60]) / yscale) << "," << ((datavalues[69] - datavalues[61]) / xscale) << ",";
			selFeatures << ((datavalues[80] - datavalues[88]) / yscale) << "," << ((datavalues[81] - datavalues[89]) / xscale) << ",";
			selFeatures << ((datavalues[64] - datavalues[72]) / yscale) << "," << ((datavalues[65] - datavalues[73]) / xscale) << ",";
			selFeatures << ((datavalues[84] - datavalues[92]) / yscale) << "," << ((datavalues[85] - datavalues[93]) / xscale) << ",";
			selFeatures << ((datavalues[42] - datavalues[104]) / yscale) << "," << ((datavalues[43] - datavalues[105]) / xscale) << ",";
			selFeatures << ((datavalues[44] - datavalues[104]) / yscale) << "," << ((datavalues[45] - datavalues[105]) / xscale) << ",";
			selFeatures << ((datavalues[42] - datavalues[44]) / yscale) << "," << ((datavalues[43] - datavalues[45]) / xscale) << ",";
			selFeatures << ((datavalues[25] - datavalues[1])/xscale) << ",";
			selFeatures << ((datavalues[12] - datavalues[28])/yscale) << ",";
			selFeatures << ((datavalues[153])) << "\n";


		}
		inlandmarks.close();
		selFeatures.close();
	}

}