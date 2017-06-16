#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

const string window_name = "xyz";

int getRandom(int ul, int ll){

   return ( rand()%ul+1 );
}


// string line is a single line of the ground truth of the dataset
void getPoints(vector<Point> &p, string line){


   int count = 0;
   int pos = 0;
   
   Point p1;

   while(count < 2){

      string num;
      
      while(line[pos] != ','){
         num += line[pos++];
      }
      // for space after every comma
      pos += 2;

      if(count == 0)
         p1.x = stoi(num);
      else if(count == 1)
         p1.y = stoi(num);
      count++;
   }
   //cout<<p1.x<<" "<<p1.y<<" ";
   p.push_back(p1);

   while(count < 4){

      string num;

      while(line[pos] != ','){
         num += line[pos++];
      }

      // for space after every comma
      pos += 2;

      if(count == 2)
         p1.x = stoi(num);
      else if(count == 3)
         p1.y = stoi(num);
      count++;
   }
   //cout<<p1.x<<" "<<p1.y<<endl;
   p.push_back(p1);
}

void readFilesPositive(string location){

	int count = 78900;
	int dim = 20;

	ofstream fout("positive.txt");

	for(int i=1; i<=count; i++){

		Mat img = imread(location+to_string(i)+".png", 0);

		if(!img.data)
			continue;

		resize(img, img, Size(20, 20));

        HOGDescriptor hog( Size(dim, dim), Size(4, 4), Size(2, 2), Size(2, 2), 2 );
        vector<float> ders;
        vector<Point>locs;
        hog.compute(img, ders, Size(0,0), Size(0,0), locs);

        for(int j=0; j<ders.size(); j++)
       		fout<<to_string(ders[j])<<" ";
       	fout<<endl;
       	cout<<i*100./count<<"%"<<endl;
	}
	fout.close();
}

void readFilesNegative(string location){

	int count = 9800;
	int dim = 20;
	ofstream fout("negative.txt");

	for(int i=1; i<=count; i++){

		Mat img = imread(location+to_string(i)+".jpg", 0);

		if(!img.data)
			continue;

		for(int j=0; j<3; j++){
	        int pointX = getRandom(img.cols-dim-1, 0);
	        int pointY = getRandom(img.rows-dim-1, 0);
	        Mat tmp(img, Rect(Point(pointX, pointY), Point(pointX+dim, pointY+dim)));
	        HOGDescriptor hog( Size(dim, dim), Size(4, 4), Size(2, 2), Size(2, 2), 2 );
	        vector<float> ders;
	        vector<Point>locs;

	        hog.compute(tmp, ders, Size(0,0), Size(0,0), locs);
	        for(int j=0; j<ders.size(); j++)
	           fout<<to_string(ders[j])<<" ";

	        fout<<endl;

	        cout<<i*100./count<<"%"<<endl;
	    }
	}
	fout.close();
}


void filterRegions(Mat &img, vector<Rect> &mser_regions, vector<Rect> &filtered_regions){


	int dim = 20;
    ofstream fout("tmp_hog.txt");
    ifstream fin("result.txt");
    namedWindow(window_name, WINDOW_NORMAL);

    for(int i=0; i<mser_regions.size(); i++){

    	Mat tmp(img, Rect(mser_regions[i]));
    	cvtColor(tmp, tmp, CV_BGR2GRAY);
    	resize(tmp, tmp, Size(dim, dim));

	    HOGDescriptor hog( Size(dim, dim), Size(4, 4), Size(2, 2), Size(2, 2), 2 );
	    vector<float> ders;
	    vector<Point>locs;

	    hog.compute(tmp, ders, Size(0,0), Size(0,0), locs);
	    for(int j=0; j<ders.size(); j++)
	    	fout<<ders[j]<<" ";
	    fout<<endl;
	}
	fout.close();
	system("python test.py");

	string line;
	int i=0;
	while(getline(fin, line)){

		if(line == "0")
			filtered_regions.push_back(mser_regions[i]);
		i++;
	}
	fin.close();
	Mat tmp = img.clone();
	cvtColor(tmp, tmp, CV_BGR2GRAY);
	for(int i=0; i<filtered_regions.size(); i++){

		rectangle(tmp, filtered_regions[i], Scalar(255), 1, 8, 0);
	}

	imshow(window_name, tmp);
	waitKey(0);
}
void getMSER(Mat &image, vector<Rect> &boxes){

   namedWindow(window_name, WINDOW_NORMAL);
   Mat mser_img;
   cvtColor(image, mser_img, CV_BGR2GRAY);
   
   // Histogram equalization/increasing contrrast does not help but blurring does
   GaussianBlur(mser_img, mser_img, Size(5, 5), 0, 0);
   //equalizeHist(mser_img, mser_img);
   imshow(window_name, mser_img);
   waitKey(0);

   
   vector<vector<Point> > contours;
   MSER ms;
   //MSER ms(10, (int)(0.00002*mser_img.cols*mser_img.rows), (int)(0.05*mser_img.cols*mser_img.rows), 1, 0.7);;
   ms(mser_img, contours, Mat());

   for(int i=0; i<contours.size(); i++){

      boxes.push_back(boundingRect(contours[i]));
      rectangle(mser_img, boxes[i], Scalar(255), 1, 8, 0);

   }

   //imwrite("mser_output.jpg", mser_img);
   imshow(window_name, mser_img);
   waitKey(0);
}

void filterByRatioAndArea(Mat &img, vector<Rect> &mser_regions){

	Mat image = img.clone();
	cvtColor(image, image, CV_BGR2GRAY);
   float image_area = image.rows*image.cols;
   groupRectangles(mser_regions, 1, 0.2);
   for(int i=0; i<mser_regions.size();i++){

   	for(int j=0; j<mser_regions.size(); j++){

   		if(i==j)
   			continue;
   		else if((mser_regions[i] & mser_regions[j]) == mser_regions[i]){
   			mser_regions.erase(mser_regions.begin()+i);
   		}
   		
   	}
   }
   for(int i=0; i<mser_regions.size();i++){

   	for(int j=0; j<mser_regions.size(); j++){

   		if(i==j)
   			continue;
   		else if((mser_regions[i] & mser_regions[j]).area() > 0){
   			Rect newRect = mser_regions[i]|mser_regions[j];
   			mser_regions.erase(mser_regions.begin()+j);
   			mser_regions.erase(mser_regions.begin()+i);
   			mser_regions.push_back(newRect);
   		}
   		}
   }
   
   for(int i=0; i<mser_regions.size(); i++){

      float ratio = ((float)mser_regions[i].width/(float)mser_regions[i].height);
      float rect_area = mser_regions[i].height*mser_regions[i].width;

      //Keep aspect ratio between 0.1 and 10
      if(ratio <0.1 || ratio > 10)
         mser_regions.erase(mser_regions.begin()+i);

      else if(rect_area/image_area < 0.00005)
         mser_regions.erase(mser_regions.begin()+i);

      else
         rectangle(image, mser_regions[i], Scalar(255), 1, 8, 0);

   }

   
   //imwrite("op.jpg", image);
   namedWindow("ratio and area", WINDOW_NORMAL);
   imshow("ratio and area", image);
   waitKey(0);
}


Point computeCenterPoint(Rect r){

	Point ans;
	ans.x = (r.x + r.x+r.width);
	ans.y = (r.y + r.y+r.height);
	return ans;
}

float getDistance(Point p1, Point p2){

	float ans = sqrt( (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) );
	return ans;
}
void removeRedundantBoxes(Mat &img, vector<Rect> &mser_regions){


	for(int i=0; i<mser_regions.size(); i++){

		Point r1_start = Point(mser_regions[i].x, mser_regions[i].y);
		Point r1_end = Point(mser_regions[i].x + mser_regions[i].width , mser_regions[i].y + mser_regions[i].height);
		for(int j=0; j<mser_regions.size(); j++){
			
			if(i == j)
				continue;
			Point r2_start = Point(mser_regions[j].x, mser_regions[j].y);
			Point r2_end = Point(mser_regions[j].x + mser_regions[j].width , mser_regions[j].y + mser_regions[j].height);

			if(r1_start.x <= r2_start.x && r1_start.y <= r2_start.y && r1_end.x >= r2_end.x && r1_end.y >= r2_end.y)
				mser_regions.erase(mser_regions.begin()+j);
		}
	}
	Mat tmp = img.clone();
	cvtColor(tmp, tmp, CV_BGR2GRAY);
	for(int i=0; i<mser_regions.size(); i++)
		rectangle(tmp, mser_regions[i], Scalar(255), 1, 8, 0);
	namedWindow("filtermore", WINDOW_NORMAL);
	imwrite("as.jpg", tmp);
	imshow("filtermore", tmp);
	waitKey(0);

}

void combineBoxes(string file_name, Mat &img, vector<Rect> &mser_regions){

	float thresh = 20.;
	bool change = false;

	while(change){
		change = false;
		for(int i=0; i<mser_regions.size(); i++){

			Point c1 = computeCenterPoint(mser_regions[i]);

			for(int j=0; j<mser_regions.size(); j++){

				if(i == j)
					continue;

				Point c2 = computeCenterPoint(mser_regions[j]);
				if(getDistance(c1, c2) <= thresh){

					Rect tmp = mser_regions[i]|mser_regions[j];
					mser_regions.erase(mser_regions.begin()+i);
					mser_regions.erase(mser_regions.begin()+j);
					mser_regions.push_back(tmp);
					change = true;
				}

			}
		}
	}
	Mat tmp = img.clone();
	cvtColor(tmp, tmp, CV_BGR2GRAY);
	for(int i=0; i<mser_regions.size(); i++)
		rectangle(tmp, mser_regions[i], Scalar(255), 1, 8, 0);
	namedWindow("combine", WINDOW_NORMAL);
	imshow("combine", tmp);
	waitKey(0);
}

void applyThresholding(Mat &img, vector<Rect> &filtered_regions){


	for(int i=0; i<filtered_regions.size(); i++){

		Mat tmp(img, filtered_regions[i]);
		cvtColor(tmp, tmp, CV_BGR2GRAY);
		adaptiveThreshold(tmp, tmp, 100, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2.0);
		imshow(window_name, tmp);
		waitKey(0);
	}
}

int main(){

	string file_name = "temp6.jpeg";
	vector<Rect> mser_regions, filtered_regions;

	string location_positive = "positive/";
	string location_negative = "negative/";
	//readFilesPositive(location_positive);
	//readFilesNegative(location_negative);
	
	
	Mat img = imread(file_name);

	getMSER(img, mser_regions);
	cout<<mser_regions.size()<<endl;
	filterRegions(img, mser_regions, filtered_regions);
	removeRedundantBoxes(img, filtered_regions);
	combineBoxes(file_name, img, filtered_regions);
}