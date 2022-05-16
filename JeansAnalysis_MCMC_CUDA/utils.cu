/*
 *
 * utility function
 *
 */

#include "functions.h"
#include <stdlib.h>
using namespace std;

float** DataArray;

//Read input file with mock data
float** read_file(std::string filename,int rows)
{
	int cols = 3;
	std::fstream file;
	file.open(filename.c_str(), std::ios::in);
	if(!file.is_open()){return 0;}

	float** floats = new float*[cols+1];
	for(int i = 0; i <cols;++i){ floats[i] = new float[rows+1]; }

	//read each row
	//cols: R_proj, vel, vel_err
	for(int i =0;i<rows;++i)
	{
		for(int j =0;j<cols;++j)//push into the col
		{ file >>floats[j][i]; }
	}
	file.close();

	return floats;
}
