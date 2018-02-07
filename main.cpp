#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

int main(){
    string orgFilePath = "./img/pencils.jpg";
    string bayerFilePath = "./img/pencils_mosaic.bmp";
    cv::Mat orgImg = cv::imread(orgFilePath.c_str());
    cv::Mat orgImgF;
    orgImg.convertTo(orgImgF, CV_32FC3);
    orgImgF *= (float)1/255;
    if(!orgImgF.empty()){
        cv::imshow("Original Image", orgImgF);
    }
    cv::Mat bayerImg = cv::imread(bayerFilePath.c_str(), 0);
    cv::Mat bayerImgF;
    bayerImg.convertTo(bayerImgF, CV_32FC1);
    bayerImgF *= (float)1/255;
//    if(!bayerImgF.empty()){
//        cv::imshow("Bayer Image", bayerImgF);
//    }
    int width = bayerImg.rows;
    int height = bayerImg.cols;

    cv::Mat blue = cv::Mat(width, height, CV_32F, float(0));
    cv::Mat green = cv::Mat(width, height, CV_32F, float(0));
    cv::Mat red = cv::Mat(width, height, CV_32F, float(0));

    float val;
    for(int i=0; i<width; i++){
        for(int j=0; j<height; j++){
            val = bayerImgF.at<float>(i, j);
            if(i%2 == 0){
                if(j%2 == 0){
                    red.at<float>(i, j) = val;
                }
                else{
                    green.at<float>(i, j) = val;
                }
            }
            else{
                if(j%2 == 0){
                    green.at<float>(i, j) = val;
                }
                else{
                    blue.at<float>(i, j) = val;
                }
            }
        }
    }

    float rbCorr[9] = {0.25,0.5,0.25, 0.5,1.0,0.5, 0.25,0.5,0.25};
    float gCorr[9] = {0.0,0.25,0.0, 0.25,1.0,0.25, 0.0,0.25,0.0};

    cv::Mat rbKernel = cv::Mat(3, 3, CV_32F, rbCorr);
    cv::Mat gKernel = cv::Mat(3, 3, CV_32F, gCorr);

    cv::Mat blueCh, greenCh, redCh;

    cv::filter2D(blue, blueCh, -1, rbKernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(green, greenCh, -1, gKernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(red, redCh, -1, rbKernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    std::vector<cv::Mat> mergeArray;
    mergeArray.push_back(blueCh);
    mergeArray.push_back(greenCh);
    mergeArray.push_back(redCh);

    cv::Mat demImg;
    cv::merge(mergeArray, demImg);
    cv::imshow("Demosaiced Image", demImg);

    cv::Mat sq1, sq2, diffImg;
    sq1 = orgImgF.mul(orgImgF);
    sq2 = demImg.mul(demImg);
    cv::absdiff(sq1, sq2, diffImg);
    cv::imshow("Squared Differences", diffImg);

// ##############################

    cv::Mat bfBlue, bfRed, medBlue, medRed;

    medBlue.convertTo(medBlue, CV_32FC1);
    medRed.convertTo(medRed, CV_32FC1);

    cv::subtract(blueCh, greenCh, bfBlue);
    cv::subtract(redCh, greenCh, bfRed);

    cv::medianBlur(bfBlue, medBlue, 5);
    cv::medianBlur(bfRed, medRed, 5);

    cv::add(medBlue, greenCh, bfBlue);
    cv::add(medRed, greenCh, bfRed);

    std::vector<cv::Mat> bfMergeArray;
    bfMergeArray.push_back(bfBlue);
    bfMergeArray.push_back(greenCh);
    bfMergeArray.push_back(bfRed);

    cv::Mat bfDemImg;
    cv::merge(bfMergeArray, bfDemImg);
    cv::imshow("BF Demosaiced Image", bfDemImg);

    cv::Mat bfSq1, bfSq2, bfDiffImg;
    bfSq1 = orgImgF.mul(orgImgF);
    bfSq2 = bfDemImg.mul(bfDemImg);
    cv::absdiff(bfSq1, bfSq2, bfDiffImg);
    cv::imshow("BF Squared Differences", bfDiffImg);

    cv::waitKey(0);
    return 0;
}
