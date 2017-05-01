#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat image;
    Mat image_color;
    Mat image_prev;
    Mat image_video;
    Mat image_diff;
    Mat image_first;
    Mat image_last;
    string filename;
    string header;
    string tail;
    Size winsize = Size(21,21);
    vector<Point2f> corners, corners_prev,corners_first,corners_filt;
    vector<Rect> roi_rect;
    vector<Mat> roi_templ;
    Size roi_size = Size(31,31);
    Mat roi_big;
    Rect roi_big_rect;
    Size search_size = Size(81,81);
    vector<Rect> roi_search_rect;
    vector<Mat> roi_search;
    Mat result;
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point roi_center, rect_search_center;
    vector<int> condition_tracker;
    Mat ransac_num;
    vector<int> remove_tracker;
    int reject_count = 0;
    vector<int> keep_tracker;
    int keep_count = 0;
    Mat F_mask;
    Mat H1, H2;
    vector<uchar> status;
    vector<float> err;

    header = "/home/dallin/robotic_vision/Project/Archery/G";
    tail = ".jpg";

    // Get the first frame features.
    image = imread("/home/dallin/robotic_vision/Project/Archery/G1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    resize(image,image,Size(480,640));
    // Define a large roi that keeps the feature corners not near the edges of the image.
    goodFeaturesToTrack(image,corners,300,.1,10,noArray(),3,false,.04);
    goodFeaturesToTrack(image,corners_first,300,.1,10,noArray(),3,false,.04);
    image_first = imread("/home/dallin/robotic_vision/Project/Archery/G1.jpg",CV_LOAD_IMAGE_COLOR);
    resize(image_first,image_first,Size(480,640));
    image_last = imread("/home/dallin/robotic_vision/Project/Archery/G4.jpg",CV_LOAD_IMAGE_COLOR);
    resize(image_last,image_last,Size(480,640));

    for(int i = 0; i<corners.size(); i++)
    {
        condition_tracker.push_back(1); //All points are marked initially as valid and current.
    }


    for (int i = 1; i < 5; i+=1)
    {
        filename = header + to_string(i) + tail;
        if(image.empty())
        {
            image = Mat::zeros(480,640,CV_32F);
        }

        image_prev = image.clone();
        corners_prev = corners;
        image = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
        resize(image,image,Size(480,640));
        image_color = imread(filename,CV_LOAD_IMAGE_COLOR);
        resize(image_color,image_color,Size(480,640));
        //        cout << corners_prev << endl;

//*********************************This is template Matching *************************************
//        // Draw templates around corners on old frame.
//        for (int i=0; i<corners_prev.size(); i++)
//        {
//            roi_center = Point(corners_prev[i].x-roi_size.width/2,corners_prev[i].y-roi_size.height/2);
//            rect_search_center = Point(corners_prev[i].x-search_size.width/2,corners_prev[i].y-search_size.height/2);

//            //            if(rect_search_center.x < 0 || rect_search_center.x > image.cols-search_size.width || )

//            roi_rect.push_back(Rect(roi_center,roi_size));
//            roi_search_rect.push_back(Rect(rect_search_center,search_size));

//            try
//            {

//                roi_search.push_back(image(roi_search_rect[i]));    //Current frame, but centered on old frame.
//                roi_templ.push_back(image_prev(roi_rect[i]));       //Previous frame.

//                // Draw rectangle around Template
//                //                rectangle(image_color,roi_rect[i],Scalar(0,0,255),1,LINE_8,0);
//                //                rectangle(image_color,roi_search_rect[i],Scalar(0,255,0),1,LINE_8,0);
//            }
//            catch (const exception& e)
//            {

//                //Remove point from array?
//                corners_prev.erase(corners_prev.begin() + i);
//                corners.erase(corners.begin() + i);
//                roi_rect.erase(roi_rect.begin() + i);
//                roi_search_rect.erase(roi_search_rect.begin() + i);
//                condition_tracker.erase(condition_tracker.begin() + i);
//                corners_first.erase(corners_first.begin() + i);
//                //                roi_templ.erase(roi_templ.begin() + i);
//                //                roi_search.erase(roi_search.begin() + i);
//                remove_tracker.push_back(i + reject_count); //Keep track of the points that were removed
//                reject_count = reject_count +1; //Since we're adjusting i, we need to add this in to get original location.

//                cout << "edge case" << endl;
//                i = i-1;
//            }
//        }

//        //Update new corners as where the template was found in the new image.
//        for(int i=0; i<roi_search.size(); i++)
//        {
//            matchTemplate(roi_search[i],roi_templ[i],result,CV_TM_SQDIFF,noArray());
//            normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
//            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
//            corners[i] = Point(minLoc.x+roi_search_rect[i].x+roi_size.width/2, minLoc.y+roi_search_rect[i].y+roi_size.height/2);
//            //            cout << minLoc << endl;
//        }

        //***************************************This is the end of Template matching. **********************************

        calcOpticalFlowPyrLK(image_prev,image,corners_prev,corners,status,err,winsize,3,
                                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,.01),0,1e-4);

        findFundamentalMat(corners_prev,corners,ransac_num,FM_RANSAC,3,.99);
        for(int i = 0; i<ransac_num.rows; i++)
        {
            condition_tracker[i] = ransac_num.at<bool>(0,i);
        }

        for(int i = 0; i<condition_tracker.size(); i++)
        {
            if (condition_tracker[i] == 0)
            {
                //Remove point from array
                corners_prev.erase(corners_prev.begin() + i);
                corners.erase(corners.begin() + i);
                roi_rect.erase(roi_rect.begin() + i);
                roi_search_rect.erase(roi_search_rect.begin() + i);
                condition_tracker.erase(condition_tracker.begin() + i);
                corners_first.erase(corners_first.begin() + i);
                remove_tracker.push_back(i + reject_count); //Keep track of the points that were removed
                reject_count = reject_count +1; //Since we're adjusting i, we need to add this in to get original location.

                cout << "edge case" << endl;
                i = i-1;
            }

        }


        for(int i=0; i<corners.size(); i++)
        {
            circle(image_color,corners_first[i],1,Scalar(0,255,0),2,LINE_8,0);
        }
        for(int i=0; i<corners.size(); i++)
        {
            line(image_color,corners[i],corners_first[i],Scalar(0,0,255),1,LINE_8,0);
        }


        //        imshow("Image",image_color);
        //        imshow("ROI",roi_templ[0]);
        //        imshow("Search",roi_search[0]);


        roi_search.clear();
        roi_templ.clear();
        roi_rect.clear();
        roi_search_rect.clear();
        waitKey(1);
    }

    Mat F = findFundamentalMat(corners_first,corners,F_mask,FM_8POINT,3,0.99);
    stereoRectifyUncalibrated(corners_first,corners,F,Size(640,480),H1,H2,5);

    Mat M1_guess = (Mat_<double>(3,3) << 1000, 0, 240, 0, 1000, 320, 0, 0, 1);
    Mat d1_guess = (Mat_<double>(5,1) << -.3, .1, 0, 0, .5);
    Mat M2_guess = (Mat_<double>(3,3) << 1000, 0, 240, 0, 1000, 320, 0, 0, 1);
    Mat d2_guess = (Mat_<double>(5,1) << -.3, .1, 0, 0, .5);
    Mat R1 = M1_guess.inv(DECOMP_LU)*H1*M1_guess;
    Mat R2 = M2_guess.inv(DECOMP_LU)*H2*M2_guess;


    Mat map1, map2, map3, map4, image_first_rect,image_last_rect;
    initUndistortRectifyMap(M1_guess,d1_guess,R1,M1_guess,Size(640,480),CV_32FC1,map1,map2);
    initUndistortRectifyMap(M2_guess,d2_guess,R2,M2_guess,Size(640,480),CV_32FC1,map3,map4);
    remap(image_first,image_first_rect,map1,map2,INTER_LINEAR,BORDER_CONSTANT);
    remap(image_last,image_last_rect,map3,map4,INTER_LINEAR,BORDER_CONSTANT);

    for(int i = 80; i < 480; i+=80)
    {
        line(image_first_rect,Point(0,i),Point(640,i),Scalar(0,255,0),1,LINE_8,0);
        line(image_last_rect,Point(0,i),Point(640,i),Scalar(0,255,0),1,LINE_8,0);
    }

//    cout << H1 << endl;
    absdiff(image_first_rect, image_last_rect,image_diff);
    imshow("Diff", image_diff);
    imwrite("ParallelRealFirst.jpg", image_first_rect);
    imwrite("ParallelReallast.jpg",image_last_rect);
    imshow("Rectified", image_first_rect);
    imshow("Last",image_last_rect);
    waitKey(0);

    return 0;
}
