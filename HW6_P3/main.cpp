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
    Mat image_first;
    Mat image_last;
    string filename;
    string header;
    string tail;
    Size winsize = Size(21,21);
    vector<Point2f> corners, corners_prev,corners_first;
    vector<Point3f> corners_diff, corners_3d;
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
    Size imageSize;

    Mat P1, P2, Q;

    float fx, fy, cx, cy;

    Point2f point_diff;
    vector<Point2f> point_diff_vec;

    header = "/home/dallin/robotic_vision/HW6/Parallel_Real/ParallelReal";
    tail = ".jpg";


    // Get the first frame features.
    image = imread("/home/dallin/robotic_vision/HW6/Parallel_Real/ParallelReal10.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    // Define a large roi that keeps the feature corners not near the edges of the image.
    goodFeaturesToTrack(image,corners,500,.01,10,noArray(),3,false,.04);
    goodFeaturesToTrack(image,corners_first,500,.01,10,noArray(),3,false,.04);
    Size winSize = Size( 5, 5 );
    Size zeroZone = Size( -1, -1 );
    TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
    cornerSubPix(image,corners,winSize,zeroZone,criteria);
    cornerSubPix(image,corners_first,winSize,zeroZone,criteria);
    image_first = imread("/home/dallin/robotic_vision/HW6/Parallel_Real/ParallelReal10.jpg",CV_LOAD_IMAGE_COLOR);
    image_last = imread("/home/dallin/robotic_vision/HW6/Parallel_Real/ParallelReal15.jpg",CV_LOAD_IMAGE_COLOR);

    for(int i = 0; i<corners.size(); i++)
    {
        condition_tracker.push_back(1); //All points are marked initially as valid and current.
    }


    for (int i = 10; i < 16; i+=1)
    {
        filename = header + to_string(i) + tail;
        if(image.empty())
        {
            image = Mat::zeros(480,640,CV_32F);
        }

        image_prev = image.clone();
        corners_prev = corners;
        image = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
        image_color = imread(filename,CV_LOAD_IMAGE_COLOR);
        //        cout << corners_prev << endl;


        // Draw templates around corners on old frame.
        for (int i=0; i<corners_prev.size(); i++)
        {
            roi_center = Point(corners_prev[i].x-roi_size.width/2,corners_prev[i].y-roi_size.height/2);
            rect_search_center = Point(corners_prev[i].x-search_size.width/2,corners_prev[i].y-search_size.height/2);

            //            if(rect_search_center.x < 0 || rect_search_center.x > image.cols-search_size.width || )

            roi_rect.push_back(Rect(roi_center,roi_size));
            roi_search_rect.push_back(Rect(rect_search_center,search_size));

            try
            {

                roi_search.push_back(image(roi_search_rect[i]));    //Current frame, but centered on old frame.
                roi_templ.push_back(image_prev(roi_rect[i]));       //Previous frame.

                // Draw rectangle around Template
                //                rectangle(image_color,roi_rect[i],Scalar(0,0,255),1,LINE_8,0);
                //                rectangle(image_color,roi_search_rect[i],Scalar(0,255,0),1,LINE_8,0);
            }
            catch (const exception& e)
            {

                //Remove point from array?
                corners_prev.erase(corners_prev.begin() + i);
                corners.erase(corners.begin() + i);
                roi_rect.erase(roi_rect.begin() + i);
                roi_search_rect.erase(roi_search_rect.begin() + i);
                condition_tracker.erase(condition_tracker.begin() + i);
                corners_first.erase(corners_first.begin() + i);
                //                roi_templ.erase(roi_templ.begin() + i);
                //                roi_search.erase(roi_search.begin() + i);
                remove_tracker.push_back(i + reject_count); //Keep track of the points that were removed
                reject_count = reject_count +1; //Since we're adjusting i, we need to add this in to get original location.

                cout << "edge case" << endl;
                i = i-1;
            }
        }

        //Update new corners as where the template was found in the new image.
        for(int i=0; i<roi_search.size(); i++)
        {
            matchTemplate(roi_search[i],roi_templ[i],result,CV_TM_SQDIFF,noArray());
            normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
            corners[i] = Point(minLoc.x+roi_search_rect[i].x+roi_size.width/2, minLoc.y+roi_search_rect[i].y+roi_size.height/2);
            //            cout << minLoc << endl;
        }

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

    Mat M1 = (Mat_<double>(3,3) <<  825.0900600547, 0, 331.6538103208,
              0, 824.2672147458,  252.9284287373,
              0, 0, 1);
    fx = 825.0900600547;
    fy = 824.2672147458;
    cx = 331.6538103208;
    cy = 252.9284287373;

    Mat d1 = (Mat_<double>(5,1) << -0.2380769337,
              0.0931325835,
              0.0003242537,
              -0.0021901930,
              0.4641735616);
    Mat M2 = (Mat_<double>(3,3) << 825.0900600547, 0, 331.6538103208,
              0, 824.2672147458,  252.9284287373,
              0, 0, 1);
    Mat d2 = (Mat_<double>(5,1) << -0.2380769337,
              0.0931325835,
              0.0003242537,
              -0.0021901930,
              0.4641735616);
    Mat R1 = M1.inv(DECOMP_LU)*H1*M1;
    Mat R2 = M2.inv(DECOMP_LU)*H2*M2;

    Mat M2_T;
    transpose(M2,M2_T);
    Mat E = M2_T*F*M1;

    Mat w,u,vt;
    SVD::compute(E,w,u,vt,0);
    w.at<double>(0) = 1;
    w.at<double>(1) = 1;
    w.at<double>(2) = 0;
    E = u*Mat::diag(w)*vt;

    Mat R, t;

    undistortPoints(corners_first,corners_first,M1,d1,noArray(),noArray());
    undistortPoints(corners,corners,M1,d1,noArray(),noArray());


    recoverPose(E,corners_first,corners,R,t,1.0,Point2d(0,0),noArray());

    Mat map1, map2, map3, map4, image_first_rect,image_last_rect;
    initUndistortRectifyMap(M1,d1,R1,M1,Size(640,480),CV_32FC1,map1,map2);
    initUndistortRectifyMap(M2,d2,R2,M2,Size(640,480),CV_32FC1,map3,map4);
    remap(image_first,image_first_rect,map1,map2,INTER_LINEAR,BORDER_CONSTANT);
    remap(image_last,image_last_rect,map3,map4,INTER_LINEAR,BORDER_CONSTANT);



    imageSize = cvSize(640,480);
    stereoRectify(M1,d1,M2,d2,imageSize,R,t,R1,R2,P1,P2,Q,CALIB_ZERO_DISPARITY,-1,imageSize,0,0);




    for(int i = 80; i < 480; i+=80)
    {
        line(image_first_rect,Point(0,i),Point(640,i),Scalar(0,255,0),1,LINE_8,0);
        line(image_last_rect,Point(0,i),Point(640,i),Scalar(0,255,0),1,LINE_8,0);
    }

    for(int i = 0; i < corners.size(); i++)
    {
        corners[i] = Point2f(corners[i].x*fx+cx, corners[i].y*fy+cy);
//        circle(image_last_rect, corners[i], 1, Scalar(255,255,255),2, LINE_8, 0);
    }

    for(int i = 0; i < corners_first.size(); i++)
    {
        corners_first[i] = Point2f(corners_first[i].x*fx+cx, corners_first[i].y*fy+cy);
//        circle(image_first, corners_first[i], 1, Scalar(0,0,255),2, LINE_8, 0);
    }

    for(int i = 0; i < corners.size(); i++)
    {
        corners_diff.push_back(Point3f(corners_first[i].x, corners_first[i].y, corners_first[i].x - corners[i].x));
    }

    perspectiveTransform(corners_diff,corners_3d,Q);
    float scale_factor = 2.56;

    for(int i = 0; i < corners_3d.size(); i++)
    {
//        corners_3d[i] = Point3f(corners_3d[i].x*scale_factor, corners_3d[i].y*scale_factor, corners_3d[i]*scale_factor);
        corners_3d[i].x = corners_3d[i].x*scale_factor;
        corners_3d[i].y = corners_3d[i].y*scale_factor;
        corners_3d[i].z = corners_3d[i].z*scale_factor;
    }

//    for(int i = 0; i < corners_first.size(); i++)
//    {
//        circle(image_first, corners_first[i], 3, Scalar(0,255,0), 3, LINE_8, 0);
//        cout << i << endl;
//        circle(image_first, corners_first[i-1], 1, Scalar(0,0,255),2, LINE_8, 0);
//        imshow("Rectified", image_first);
//        waitKey(0);
//    }

    circle(image_first, corners_first[89], 2, Scalar(0,255,0), 2, LINE_8, 0);
    circle(image_first, corners_first[0], 2, Scalar(0,255,0), 2, LINE_8, 0);
    circle(image_first, corners_first[19], 2, Scalar(0,255,0), 2, LINE_8, 0);
    circle(image_first, corners_first[154], 2, Scalar(0,255,0), 2, LINE_8, 0);

    putText(image_first, "(" + to_string(corners_3d[89].x) + ", " + to_string(corners_3d[89].y) + ", " + to_string(corners_3d[89].z)
            + ")",Point(corners_first[89].x + 3, corners_first[89].y-5), CV_FONT_HERSHEY_COMPLEX, .5, Scalar(255,255,255),1,LINE_8, false);

    putText(image_first, "(" + to_string(corners_3d[0].x) + ", " + to_string(corners_3d[0].y) + ", " + to_string(corners_3d[0].z)
            + ")",Point(corners_first[0].x - 200, corners_first[0].y + 15), CV_FONT_HERSHEY_COMPLEX, .5, Scalar(255,255,255),1,LINE_8, false);

    putText(image_first, "(" + to_string(corners_3d[19].x) + ", " + to_string(corners_3d[19].y) + ", " + to_string(corners_3d[19].z)
            + ")",Point(corners_first[19].x + 3, corners_first[19].y), CV_FONT_HERSHEY_COMPLEX, .5, Scalar(255,255,255),1,LINE_8, false);

    putText(image_first, "(" + to_string(corners_3d[154].x) + ", " + to_string(corners_3d[154].y) + ", " + to_string(corners_3d[154].z)
            + ")",Point(corners_first[154].x - 200, corners_first[154].y-5), CV_FONT_HERSHEY_COMPLEX, .5, Scalar(255,255,255),1,LINE_8, false);


    cout << t << endl;
    imshow("First", image_first);
//    imwrite("TurnedReal.jpg",image_first);
//    imshow("Last",image_last_rect);
    waitKey(0);

    return 0;
}
