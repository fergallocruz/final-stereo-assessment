#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <json/json.h>

#include <stdio.h>

using namespace cv;
// Defining callback functions for the trackbars to update parameter values
Ptr<StereoBM> bm = StereoBM::create(16, 9);
Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
int numberOfDisparities = 64;
int SADWindowSize = 255;
int preFilterType = 1;
int preFilterSize = 1;
int preFilterCap = 31;
int minDisparity = 0;
int textureThreshold = 10;
int uniquenessRatio = 15;
int speckleRange = 0;
int speckleWindowSize = 0;
int disp12MaxDiff = -1;
int dispType = CV_16S;
static void on_trackbar1(int, void*)
{
    sgbm->setNumDisparities(numberOfDisparities * 16);
    bm->setNumDisparities(numberOfDisparities * 16);
    numberOfDisparities = numberOfDisparities * 16;
}

static void on_trackbar2(int, void*)
{
    sgbm->setBlockSize(SADWindowSize * 2 + 5);
    bm->setBlockSize(SADWindowSize * 2 + 5);
    SADWindowSize = SADWindowSize * 2 + 5;
}

static void print_help_stereo()
{
    printf("\nDemo stereo matching converting L and R images into disparity "
           "and point clouds\n");
    printf("\nUsage: stereo_match <left_image> <right_image> "
           "[--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] "
           "[-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display] [-o=<disparity_image>] [-p=<point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for (int y = 0; y < mat.rows; y++)
    {
        for (int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
                continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}
int stereo_match()
{
    Json::Value root;

    std::ifstream file("settings.json");
    file >> root;

    std::string img1_filename = root["files"]["left"].asString();
    std::string img2_filename = root["files"]["right"].asString();
    std::string intrinsic_filename = root["files"]["intrinsic"].asString();
    std::string extrinsic_filename = root["files"]["extrinsic"].asString();
    std::string disparity_filename = root["files"]["disparity"].asString();
    std::string point_cloud_filename = root["files"]["point_cloud"].asString();
    std::string _alg = root["algorithm"].asString();

    enum
    {
        STEREO_BM = 0,
        STEREO_SGBM = 1,
        STEREO_HH = 2,
        STEREO_VAR = 3,
        STEREO_3WAY = 4
    };
    int alg = STEREO_SGBM;
    alg = _alg == "bm"         ? STEREO_BM
          : _alg == "sgbm"     ? STEREO_SGBM
          : _alg == "hh"       ? STEREO_HH
          : _alg == "var"      ? STEREO_VAR
          : _alg == "sgbm3way" ? STEREO_3WAY
                               : -1;

    SADWindowSize = root["blocksize"].asInt();
    numberOfDisparities = root["max-disparity"].asInt();
    bool no_display = root["no-display"].asBool();
    float scale = root["scale"].asFloat();

    if (root["help"].asBool())
    {
        print_help_stereo();
        return 0;
    }

    if (alg < 0)
    {
        printf("Command-line parameter error: Unknown stereo algorithm\n\n");
        print_help_stereo();
        return -1;
    }
    if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
    {
        printf("Command-line parameter error: The max disparity "
               "(--maxdisparity=<...>) must be a positive integer divisible by "
               "16\n");
        print_help_stereo();
        return -1;
    }
    if (scale < 0)
    {
        printf("Command-line parameter error: The scale factor (--scale=<...>) "
               "must be a positive floating-point number\n");
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    {
        printf("Command-line parameter error: The block size "
               "(--blocksize=<...>) must be a positive odd number\n");
        return -1;
    }
    if (img1_filename.empty() || img2_filename.empty())
    {
        printf("Command-line parameter error: both left and right images must "
               "be specified\n");
        return -1;
    }
    if ((!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()))
    {
        printf("Command-line parameter error: either both intrinsic and "
               "extrinsic parameters must be specified, or none of them (when "
               "the stereo pair is already rectified)\n");
        return -1;
    }

    if (extrinsic_filename.empty() && !point_cloud_filename.empty())
    {
        printf("Command-line parameter error: extrinsic and intrinsic "
               "parameters must be specified to compute the point cloud\n");
        return -1;
    }

    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input "
               "image file\n");
        return -1;
    }
    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input "
               "image file\n");
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

        if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();
    // Creating a named window to be linked to the trackbars
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity", img_size);

    // Creating trackbars to dynamically update the StereoBM parameters
    cv::createTrackbar("numDisparities", "disparity", &numberOfDisparities, 64, on_trackbar1);
    cv::createTrackbar("blockSize", "disparity", &SADWindowSize, 50, on_trackbar2);

    Rect roi1, roi2;
    Mat Q;

    if (!intrinsic_filename.empty())
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if (!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if (!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return -1;
        }
        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, img_size );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
    }
    Mat disp, disp8;
    while (true)
    {
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img1.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    if(alg==STEREO_HH)
        sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_3WAY)
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

    Mat disp, disp8;
    int64 t = getTickCount();
    if( alg == STEREO_BM )
        bm->compute(img1, img2, disp);
    else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY )
        sgbm->compute(img1, img2, disp);
    t = getTickCount() - t;
   // printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);
        if (!no_display)
        {
            namedWindow("left", cv::WINDOW_AUTOSIZE);
            imshow("left", img1);
            namedWindow("right", cv::WINDOW_AUTOSIZE);
            imshow("right", img2);
            namedWindow("disparity", cv::WINDOW_AUTOSIZE);
            imshow("disparity", disp);
        }
    }
    if (!disparity_filename.empty())
        imwrite(disparity_filename, disp8);

    if (!point_cloud_filename.empty())
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, true);
        saveXYZ(point_cloud_filename.c_str(), xyz);
        printf("\n");
    }

    return 0;
}
