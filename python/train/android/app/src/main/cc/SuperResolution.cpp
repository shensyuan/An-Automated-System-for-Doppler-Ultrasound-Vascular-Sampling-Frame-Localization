/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SuperResolution.h"

#include <android/log.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace tflite {
namespace examples {
namespace superresolution {
// TODO: make it changeable in the UI
constexpr int kThreadNum = 4;

SuperResolution::SuperResolution(const void* model_data, size_t model_size,
                                 bool use_gpu) {
  // Load the model
  model_ = TfLiteModelCreate(model_data, model_size);
  if (!model_) {
    LOGE("Failed to create TFLite model");
    return;
  }

  // Create the interpreter options
  options_ = TfLiteInterpreterOptionsCreate();

  // Choose CPU or GPU
  if (use_gpu) {
    delegate_ = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
    TfLiteInterpreterOptionsAddDelegate(options_, delegate_);
  } else {
    TfLiteInterpreterOptionsSetNumThreads(options_, kThreadNum);
  }

  // Create the interpreter
  interpreter_ = TfLiteInterpreterCreate(model_, options_);
  if (!interpreter_) {
    LOGE("Failed to create TFLite interpreter");
    return;
  }
}

SuperResolution::~SuperResolution() {
  // Dispose of the model and interpreter objects
  if (interpreter_) {
    TfLiteInterpreterDelete(interpreter_);
  }
  if (delegate_) {
    TfLiteGpuDelegateV2Delete(delegate_);
  }
  if (options_) {
    TfLiteInterpreterOptionsDelete(options_);
  }
  if (model_) {
    TfLiteModelDelete(model_);
  }
}

bool SuperResolution::IsInterpreterCreated() {
  if (!interpreter_) {
    return false;
  } else {
    return true;
  }
}
// 前處理
int** SuperResolution::mat2int(cv::Mat src){
    int **dst = new int*[src.rows];
    for(int i = 0 ;i < src.rows; ++i) {
        dst[i] = new int[src.cols];
        for(int j = 0; j < src.cols; ++j) {
            dst[i][j] =src.at<uchar>(i,j);
        }
    }
    return dst;
}//把mat 的中的灰階直轉到一個int

cv::Mat SuperResolution::int2mat(int** src, int rows, int cols){
    cv::Mat dst = cv::Mat(rows, cols,CV_8UC1);
    for (int i = 0; i< rows; i++){
        for (int j = 0; j< cols; j++){
            dst.at<uchar>(i,j) = src[i][j];
        }
    }
    return dst;
}//把int 的中的灰階直轉到mat

// 把1D int array 根據 height width 疊成2D return
int ** SuperResolution::oneDtotwoD(int * img_1D, int height, int width){
    int ** img_2D = new int *[height];
    for (int h = 0; h < height; h++){
      img_2D[h] = new int [width];
      for (int w = 0; w < width; w++){
        img_2D[h][w] = img_1D[h*width+w];
      }
    }
    return img_2D;
}

// 把2D int array 根據 height width 壓成1D return
int * SuperResolution::twoDtooneD(int ** img_2D, int height, int width){
  int * img_1D = new int [height*width];
  for(int h = 0; h < height; h++){
    for(int w = 0; w < width; w++){
      img_1D[h*width+w] = img_2D[h][w];
    }
  }
  return img_1D;
}

// 丟入2D array 根據左右上下邊界裁切 return 裁切後的2D array
int ** SuperResolution::InitCrop(int ** oriImg, int cropX0,int cropX1, int cropY0, int cropY1){
  int height = cropY1-cropY0;
  int width = cropX1-cropX0;
  int ** croppedImg = new int*[height];
  for (int h = 0; h < height; h++){
    croppedImg[h] = new int [width];
    for (int w = 0; w < width; w++){
      croppedImg[h][w] = oriImg[h+cropY0][w+cropX0];
    }
  }
  return croppedImg;
}

float * SuperResolution::smooth(float* sum, int size, int k){
    float * result = new float [size];
    int weight_sum = k*k;

  for (int i = k-1; i < size-k+1; i++){
    float temp_for_sum = 0;
    for (int w = 1; w < k; w++){
      temp_for_sum += (sum[i-w]*(k-w))/weight_sum;
      temp_for_sum += (sum[i+w]*(k-w))/weight_sum;
    }
    temp_for_sum += (sum[i]*k)/weight_sum;
    result[i] = temp_for_sum;
  }
  return result;
}

int * SuperResolution::get_cropImg_axis(int ** img, int height, int width){
  //  return: cropLineUp, axis, cropLineDown
  int * result = new int [3];
  float * row_sum = new float [height];
  for (int h = 0; h < height; h++){
    int sum = 0;
    for (int w = 0; w < width; w++){
        sum += img[h][w];
    }
    row_sum[h] = static_cast<float>(sum);
  }
  float * row_smooth = smooth(row_sum, height, 5);
  row_smooth = smooth(row_smooth, height, 7);
  row_smooth = smooth(row_smooth, height, 15);


  std::vector<float> localMax;
  for (int i = 0; i < height; i++){
    if(i != 0 && i != height-1){
      if(row_smooth[i-1] < row_smooth[i] && row_smooth[i] > row_smooth[i+1]) {
        localMax.push_back(row_smooth[i]);
      }
    }
  }
  sort(localMax.begin(), localMax.end(), std::greater<float>());

  int wallUp, wallDown, axis;
// 如果local maximum的數量>=3，就取出值前三大的local maximum，假設為血管壁位置
  if (localMax.size()>=3) {
      float max1 = localMax[0];
      float max2 = localMax[1];
      float max3 = localMax[2];
      int wall1, wall2, wall3;
    for(int i = 0; i < height; i++) {
      if (row_smooth[i] == max1) wall1 = i;
      else if(row_smooth[i] == max2) wall2 = i;
      else if(row_smooth[i] == max3) wall3 = i;
    }
  // 上下血管壁位置取中間即為血管軸
  // 算出三組可能為血管軸的位置，找出intensity value最小者
  // 並記錄下此狀況下，血管壁的位置
    int axis1 = (wall1 + wall2)/2;
    int axis2 = (wall2 + wall3)/2;
    int axis3 = (wall1 + wall3)/2;
    axis = axis1;
    wallUp = wall1;
    wallDown = wall2;
    if (row_smooth[axis] > row_smooth[axis2]) {
      axis = axis2;
      wallUp = wall2;
      wallDown = wall3;
    }
    if (row_smooth[axis] > row_smooth[axis3]) {
      axis = axis3;
      wallUp = wall1;
      wallDown = wall3;
    }
  }

// 如果local maximum的數量>=2，兩血管壁位置取中間即為血管軸
  else if(localMax.size()==2) {
        float max1 = localMax[0];
        float max2 = localMax[1];
    int wall1, wall2;
    for(int i = 0; i < height; i++) {
      if (row_smooth[i] == max1) wall1 = i;
      else if (row_smooth[i] == max2) wall2 = i;
    }
    axis = (wall1 + wall2)/2;
    wallUp = wall1;
    wallDown = wall2;
  }

  else {
    // 無意義，確保萬一到error的時候不會掛掉
    std::cout << "error";
    result[0] = 0;
    result[1] = 1;
    result[2] = 2;
    return result;
  }

  if (wallUp>wallDown) {
      int temp = wallDown;
      wallDown = wallUp;
      wallUp = temp;
  }

// 透過血管壁與軸的位置做裁切，裁切邊界必須在img內
  int cropLineUp = wallUp-(axis-wallUp);
  if (cropLineUp<0) cropLineUp = 0;
  int cropLineDown = wallDown+(wallDown-axis);
  if (cropLineDown>height) cropLineDown = height;

//  回傳裁切軸
  result[0] = cropLineUp;
  result[1] = axis;
  result[2] = cropLineDown;
  return result;
}

int ** SuperResolution::pasteBack(int** top, int** bot, int cropUp, int axis, int cropDown, int oriHeight, int oriWidth){
    int ** paste = new int *[oriHeight];
    for (int h = 0; h < oriHeight; h++){
        paste[h] = new int [oriWidth];
        // black area
        if(h < cropUp || h >= cropDown){
            for (int w = 0; w < oriWidth; w++){
                paste[h][w] = 0;
            }
        }
        // top
        else if(h < axis){
            for (int w = 0; w < oriWidth; w++){
                paste[h][w] = top[h-cropUp][w];
            }
        }
        // bot
        else{
            for (int w = 0; w < oriWidth; w++){
                paste[h][w] = bot[h-axis][w];
            }
        }
    }
    return paste;

}

cv::Mat SuperResolution::eliNoise(cv::Mat src, int dark, int percentage){
    std::vector<int> notDark;
    for (int h = 0; h < src.rows; h++){
        for (int w = 0; w < src.cols; w++){
            if(src.at<uchar>(h,w) > dark){
                notDark.push_back(src.at<uchar>(h,w));
            }
        }
    }
    sort(notDark.begin(), notDark.end());
    int threshold = notDark[std::round(notDark.size()*percentage/100)];

//    multiply = 255/(255-minus)
//    img = np.where(img >= minus, (img-minus)*multiply, 0)
    float multiply = 255/(255-threshold);
    for (int h = 0; h < src.rows; h++){
        for (int w = 0; w < src.cols; w++){
            if(src.at<uchar>(h,w) >= threshold){
                src.at<uchar>(h,w) = std::round((src.at<uchar>(h,w)-threshold)*multiply);
            }
            else{
                src.at<uchar>(h,w) = 0;
            }
        }
    }
    return src;
}
cv::Mat SuperResolution::imgsToPrewitt(cv::Mat srcImage){
    srcImage.convertTo(srcImage, CV_64F);

    int ddepth = -1;
    cv::Point anchor = cv::Point(-1, -1);
    cv::Mat Prewitt, grad_x, grad_y;
    cv::Mat kernelx = (cv::Mat_<double>(3,3) << -1., -1., -1., 0., 0., 0., 1., 1., 1.);
    cv::Mat kernely = (cv::Mat_<double>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);

    // OpenCV convolution
    cv::filter2D(srcImage, grad_x, ddepth, kernelx, anchor, cv::BORDER_CONSTANT);
    cv::filter2D(srcImage, grad_y, ddepth, kernely, anchor, cv::BORDER_CONSTANT);

    grad_x.convertTo(grad_x, CV_8UC1);
    grad_y.convertTo(grad_y, CV_8UC1);
    for(size_t i = 0; i < grad_y.rows; ++i) {
        for(size_t j = 0; j < grad_y.cols; ++j) {
            grad_y.at<uchar>(i, j) =255-( grad_y.at<uchar>(i, j) - grad_x.at<uchar>(i, j));
        }
    }

    return grad_y;
}
//前處理區塊結束

// resize到模型輸入大小，丟入模型，取出模型輸出，根據閾值轉成binary image return 
int SuperResolution::doseg(cv::Mat src, int** out,TfLiteInterpreter* interpreter_ , bool istop){
    if(istop){
        cv::flip(src, src, 0);//0 是上下翻轉
    }
//    cv::Mat bilateral_img = src.clone();
//    bilateralFilter(src, bilateral_img, 9, 50, 50);
    cv::Mat sobel_img = src.clone();

  
  cv::Sobel(src, sobel_img, CV_32F,0,1);
  cv::resize(sobel_img, sobel_img,cv::Size(512,128), 0, 0, cv::INTER_AREA);

  cv::Mat prewitt_img = imgsToPrewitt(src);
  prewitt_img.convertTo(prewitt_img, CV_32F);
  cv::resize(prewitt_img, prewitt_img,cv::Size(512,128), 0, 0, cv::INTER_AREA);//prewitt

  TfLiteTensor* input_tensor =
          TfLiteInterpreterGetInputTensor(interpreter_, 0);

  // Extract RGB values from each pixel
  float input_buffer[modelinputHeight* modelinputWidth* modelinputChannels];
  
  for (int i = 0, k = 0; i < 128; i++) {
    for(int j = 0; j < 512; j++){
      input_buffer[k++] = static_cast<float>(sobel_img.at<float>(i,j)/255);
      input_buffer[k++] = static_cast<float>(prewitt_img.at<float>(i,j)/255);
    }
  }
  // Feed input into model
    TfLiteStatus status = TfLiteTensorCopyFromBuffer(
          input_tensor, input_buffer,
          modelinputHeight* modelinputWidth* modelinputChannels* sizeof(float));
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when copying input buffer to input tensor");
    return -1;
  }
  // Run the interpreter
  status = TfLiteInterpreterInvoke(interpreter_);
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when running the TFLite model");
    return -1;
  }

  // Extract the output tensor data
  const TfLiteTensor* output_tensor =
          TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    float output_buffer[modeloutputHeight* modeloutputWidth* modeloutputChannel];
  status = TfLiteTensorCopyToBuffer(
          output_tensor, output_buffer,
          modeloutputHeight* modeloutputWidth* modeloutputChannel * sizeof(float));
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when copying output tensor to output buffer");
    return -1;
  }
  for(int i = 0, k= 0; i< 128; i++){
    for(int j = 0; j<  512; j++){
      out[i][j] = (output_buffer[k++]*255 > outputthreshold)? 255:0; 
    }
  }
  return 0;

}

cv::Mat SuperResolution::postprocess(cv::Mat src, cv::Mat osrc,int croplinemid , int croplineup)
{
    //cvtColor(src, src, cv::COLOR_RGB2GRAY);
    cv::cvtColor(osrc, osrc,cv::COLOR_GRAY2BGR);
    cv::Mat thresh;
    const int img_width = inputWidth;//src.cols;
    const int img_height = inputHeight;//src.rows;
    const int oimg_width = inputWidth;//osrc.cols;
    const int oimg_height = inputHeight;//osrc.rows;
    int flex = 10;
    int th = outputthreshold;
    float average_imt = 0;
    int count = 0;
    cv::threshold(src, thresh, th, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresh, contours,hierarchy,cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE); // Find the contours in the image
    cv::Mat Contours = cv::Mat::zeros(src.size(),CV_8UC3);
    int roi[50]={},rect2[50][4]={},imt[50]={},upper_index[50]={},bottom_index[50]={};
    int bound[4]={0,img_height,0,0};
    std::vector<cv::Rect> boundRect(1);
    for(int i = 0; i < contours.size();++i){
        if(contourArea(contours[i]) > 150){
            boundRect[0] = boundingRect(contours[i]);
            if (boundRect[0].y < croplinemid &&  boundRect[1].y > bound[0]){
                bound[0] = boundRect[0].y;
                bound[2] = boundRect[0].y+boundRect[0].height;
            }else if (boundRect[0].y + boundRect[0].height > croplinemid && boundRect[0].y+boundRect[0].height < bound[1]){
                bound[1] = boundRect[0].y;
                bound[3] = boundRect[0].y+boundRect[0].height;
            }
        }
    }
    for(int i = 0; i < contours.size();++i){
        if(contourArea(contours[i]) > 150){
            boundRect[0] = boundingRect(contours[i]);
            if (boundRect[0].y < croplinemid && (boundRect[0].y > bound[0] - flex)||(boundRect[0].y+boundRect[0].height > bound[0]-5) && (boundRect[0].y < bound[2])){
                drawContours(Contours, contours, i, cv::Scalar(0,0,255), 2);
                roi[i] = 1;
                upper_index[i]=1;
            }else if(boundRect[0].y + boundRect[0].height > croplinemid && (boundRect[0].y+boundRect[0].height < bound[3] + flex)or(boundRect[0].y<bound[3]+5) && (boundRect[0].y+boundRect[0].height > bound[1])){
                drawContours(Contours, contours, i, cv::Scalar(0,0,255), 2);
                roi[i] = 1;
                bottom_index[i]=1;
            }
        }
    }
    for(int i = 0;i < contours.size();++i){
        if(roi[i] == 1){
            boundRect[0] = boundingRect(contours[i]);
            rect2[i][0] = boundRect[0].x;
            rect2[i][1] = boundRect[0].y;
            rect2[i][2] = boundRect[0].width;
            rect2[i][3] = boundRect[0].height;
            imt[i] = boundRect[0].y + boundRect[0].height;
        }
    }
    for(int i = 0;i < contours.size();++i){
        if(roi[i] == 1){
            if(imt[i] > croplinemid){
                average_imt += contourArea(contours[i]);
                count += rect2[i][2];
            }
        }
    }
    average_imt /= count;

    float average_up_imt = 0;
    int count2 = 0;
    for(int i = 0;i < contours.size();++i){
        if(roi[i] == 1){
            if(imt[i] < croplinemid){
                average_up_imt += contourArea(contours[i]);
                count2 += rect2[i][2];
            }
        }
    }
    average_up_imt /= count2;
    float c[img_width][4] = {};
    float averageRatio = 0,medianRatio = 0,minRatio = 1;
    float Ratio[img_width]={};
    float median_Ratio[img_width]={};
    int len_upper = 0, len_bottom = 0;
    for(int i = 0; i < contours.size(); ++i){
            if(upper_index[i] == 1) {
                len_upper += 1;
            }
            if(bottom_index[i] == 1){
                len_bottom += 1;
            }
    }
    if(len_upper < 1){
        std::cout << "No upper boundary was labeled!\n";    //上部沒有標記到，則不算Ratio，只算IMT，Ratio在後面也不會print出來
    }
    else{
        for(int u = 0; u < contours.size();++u){
            if(upper_index[u] == 1){
                boundRect[0] = boundingRect(contours[u]);
                int temp = boundRect[0].y+boundRect[0].height;
                for(int j = boundRect[0].x; j < boundRect[0].x +boundRect[0].width; ++j){
                    for(int i = boundRect[0].y; i < boundRect[0].y +boundRect[0].height; ++i){
                        if(thresh.at<uchar>(i,j)<thresh.at<uchar>(i+1,j)){
                            c[j][0] = i;
                            break;
                        }
                    }
                    for (int i = 0; i < boundRect[0].height; ++i){
                        if(thresh.at<uchar>(temp - i,j)<thresh.at<uchar>(temp-1-i,j)){
                            c[j][1] = temp - i;
                            break;
                        }
                    }
                }
            }
        }
        for(int u = 0; u < contours.size();++u){
            if(bottom_index[u] == 1){
                boundRect[0] = boundingRect(contours[u]);
                int temp = boundRect[0].y+boundRect[0].height;
                for(int j = boundRect[0].x; j < boundRect[0].x +boundRect[0].width; ++j){
                    for(int i = boundRect[0].y; i < boundRect[0].y +boundRect[0].height; ++i){
                        if(thresh.at<uchar>(i,j)<thresh.at<uchar>(i+1,j)){
                            c[j][2] = i;
                            break;
                        }
                    }
                    for (int i = 0; i < boundRect[0].height; ++i){
                        if(thresh.at<uchar>(temp - i,j)<thresh.at<uchar>(temp-1-i,j)){
                            c[j][3] = temp - i;
                            break;
                        }
                    }
                }
            }
        }
        count = 0;
        for(int i = 0; i < img_width;++i){
            if(c[i][0]>0 && c[i][1]>0 && c[i][2]>0 && c[i][3]>0){
                Ratio[i] = (c[i][2] - c[i][1])/(c[i][3] - c[i][0]);
            }
        }
        //sum ratio
        float sumRatio = 0;
        int count_median = 0;
        for(int i = 0; i < img_width;++i){
            if(Ratio[i] > 0){
                median_Ratio[count_median++] = Ratio[i];
                sumRatio += Ratio[i];
            }
        }
        //len ratio
        float lenRatio = 0;
        int count = 0;
        for(int i = 0; i < img_width;++i){
            if(Ratio[i] > 0){
                lenRatio += 1;
            }
        }
        averageRatio = sumRatio/lenRatio;
        //median Ratio
        float temp;
        for(int i=0;i<count_median-1;++i) {
            for(int j=0;j<count_median-i-1;++j) {
                if(median_Ratio[j]>median_Ratio[j+1]){
                    temp = median_Ratio[j];
                    median_Ratio[j] = median_Ratio[j+1];
                    median_Ratio[j+1] = temp;
                }
            }
            medianRatio = median_Ratio[count_median/2];
        }

        //minRatio
        for(int i = 0; i < img_width;++i){
            if(Ratio[i] > 0){
                if(Ratio[i] < minRatio){
                    minRatio = Ratio[i];
                }
            }
        }
    }
    cv::Mat image_cropped = cv::Mat::zeros(osrc.size(),CV_8UC3);
    cv::addWeighted(osrc,0.8,Contours,1.0,0,image_cropped);  //add weught of oringinal image and contours

    /*----------------------------------
                Print part
    Show (括號後面為變數名稱)
    1. IMT (float averageIMT，若只標記到上半部則是用float average_up_imt)
    2. average of LD/IAD (float averageRatio)
    3. min of LD/IAD (float minRatio)
    4. median of LD/IAD (float medianRatio)
    這4個變數是我們希望最後可以秀出來的數字





    /*********************************
            將文字秀在圖片上
        output image(Mat text_image)
    *********************************/
    cv::Mat text_image = image_cropped.clone();
    if(len_upper > 0 && len_bottom > 0){
        std::string text("Average Ratio = "+std::to_string(averageRatio));
        std::string text1("Min Ratio     = "+std::to_string(minRatio));
        std::string text2("Median Ratio  = "+std::to_string(medianRatio));
        std::string text3("Average IMT   = "+std::to_string(average_imt)+"(pixels)");
        putText(text_image,text, cv::Point(0,img_height-55), cv::FONT_HERSHEY_SIMPLEX,0.4, cv::Scalar(0,255,255),1);
        putText(text_image,text1,cv::Point(0,img_height-40),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        putText(text_image,text2,cv::Point(0,img_height-25),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        putText(text_image,text3,cv::Point(0,img_height-10),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
    }else{   //上部or下部沒標記到IMT正常印其它NaN
        std::string text("Average Ratio = NAN");
        std::string text1("Min Ratio     = NaN");
        std::string text2("Median Ratio  = NaN");
        std::string text3("Average IMT   = "+std::to_string(average_imt)+"(pixels)");
        std::string text4("Average IMT   = "+std::to_string(average_up_imt)+"(pixels)");
        std::string text5("Average IMT   = NaN");
        putText(text_image,text,cv::Point(0,img_height-55),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        putText(text_image,text1,cv::Point(0,img_height-40),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        putText(text_image,text2,cv::Point(0,img_height-25),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        if(len_upper < 1 && len_bottom > 0){
            putText(text_image,text3,cv::Point(0,img_height-10),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        }else if(len_bottom < 1 && len_upper > 0){
            putText(text_image,text4,cv::Point(0,img_height-10),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        }else{
            putText(text_image,text5,cv::Point(0,img_height-10),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,255),1);
        }
    }

    /*----------------------------------
            OpenCV Show image part
    output image type of image_cropped => Mat
    將數據加在image_cropped上的圖 (Mat image_cropped)
    若沒有要印出來則下兩行可以刪掉
    ------------------------------------*/


    /*
        要附有文字的圖片的話用 text_image這張圖
        反之則是使用 image_cropped
    */
    //cvtColor(text_image, text_image, cv::COLOR_RGB2GRAY);
    return text_image;
}

//主函式
std::unique_ptr<int[]> SuperResolution::DoSuperResolution(int* lr_img_rgb) {
  // Allocate tensors and populate the input tensor data
//  TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
//  if (status != kTfLiteOk) {
//    LOGE("Something went wrong when allocating tensors");
//    return nullptr;
//  }
//
//  TfLiteTensor* input_tensor =
//      TfLiteInterpreterGetInputTensor(interpreter_, 0);

//  // Extract RGB values from each pixel
//  float input_buffer[kNumberOfInputPixels * kImageChannels];
//  for (int i = 0, j = 0; i < kNumberOfInputPixels; i++) {
//    // Alpha is ignored
//    input_buffer[j++] = static_cast<float>((lr_img_rgb[i] >> 16) & 0xff);
//    input_buffer[j++] = static_cast<float>((lr_img_rgb[i] >> 8) & 0xff);
//    input_buffer[j++] = static_cast<float>((lr_img_rgb[i]) & 0xff);
//  }

/////////////////////////////////////////////////////////////////////

// 因為原圖就是黑白的 所以24bit的最右邊8bit extract出來就可以
  int  Gray_input_buffer[inputPixelNumber];
  for (int i = 0; i < inputPixelNumber; i++) {
      Gray_input_buffer[i] = (lr_img_rgb[i] >> 16) & 0xff;
  }

  // 轉成2D
  int ** img_2D = oneDtotwoD(Gray_input_buffer, inputHeight, inputWidth);




/////////////////////////////////////////////////////////////////////
  int ** initCrop = InitCrop(img_2D, initCropX0, initCropX1, initCropY0, initCropY1);
  int * cropAxis = get_cropImg_axis(initCrop, initCropHeight, initCropWidth);
  int ** top = InitCrop(initCrop, 0, initCropX1 - initCropX0, cropAxis[0], cropAxis[1]);
  int ** bot = InitCrop(initCrop, 0, initCropX1 - initCropX0, cropAxis[1], cropAxis[2]);

  // 將 top bot做 eli noise(目前不含bilateral)


  // 轉成mat做後續動作，再轉回int 2d array
  cv::Mat top_mat = int2mat(top, cropAxis[1] - cropAxis[0], initCropWidth);
  cv::Mat bot_mat = int2mat(bot, cropAxis[2] - cropAxis[1], initCropWidth);

  cv::Mat top_bilateral = top_mat.clone(); //bilateral src和dst不能一樣
  cv::Mat bot_bilateral = bot_mat.clone();
  bilateralFilter(top_mat, top_bilateral, 9, 50, 50);
  bilateralFilter(bot_mat, bot_bilateral, 9, 50, 50);

  top_mat = eliNoise(top_bilateral, 20, 60);
  bot_mat = eliNoise(bot_bilateral, 20, 15);
  TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
  if (status != kTfLiteOk) {
    LOGE("Something went wrong when allocating tensors");
    return nullptr;
  }
  
  //input to model
  //top image
  int **top_out = new int*[modeloutputHeight];
  for (int i = 0; i< modeloutputHeight; i++) {
      top_out[i] = new int[modeloutputWidth];
    }
  int seg_status = doseg(top_mat, top_out, interpreter_, 1);
  if (seg_status == -1)
      return nullptr;
  //bottom image
  int **bot_out = new int*[modeloutputHeight];
  for (int i = 0; i< modeloutputHeight; i++) {
      bot_out[i] = new int[modeloutputWidth];
  }
  seg_status = doseg(bot_mat, bot_out, interpreter_, 0);
  if (seg_status == -1)
      return nullptr;

  //convert to mat to resize
  cv::Mat top_out_mat = int2mat(top_out, modeloutputHeight, modeloutputWidth);
  cv::flip(top_out_mat, top_out_mat, 0);
  cv::resize(top_out_mat, top_out_mat,cv::Size(initCropX1 - initCropX0 , cropAxis[1] - cropAxis[0]), 0, 0, cv::INTER_NEAREST);


  cv::Mat bot_out_mat = int2mat(bot_out, modeloutputHeight, modeloutputWidth);
  cv::resize(bot_out_mat, bot_out_mat,cv::Size(initCropX1 - initCropX0 , cropAxis[2] - cropAxis[1]), 0, 0, cv::INTER_NEAREST);


  int **top_result = mat2int(top_out_mat);
  int **bot_result = mat2int(bot_out_mat);

  // 貼回去最後的result
  int ** paste = pasteBack(top_result, bot_result, cropAxis[0], cropAxis[1], cropAxis[2], initCropHeight, initCropWidth);

  //cv::Mat paste_mat = int2mat(paste, initCropHeight,  initCropWidth);
    cv::Mat paste_mat = cv::Mat(inputHeight, inputWidth,CV_8UC1);
    for(int i = 0; i< inputHeight; i++){
        for(int j = 0; j< inputWidth; j++){
            paste_mat.at<uchar>(i, j) = 0;
        }
    }//initialize
    for(int i = 0; i< initCropHeight; i++){
        for(int j = 0; j< initCropWidth; j++){
            paste_mat.at<uchar>(i + initCropY0, j + initCropX0) = (uint8_t)paste[i][j];
        }
    }

  cv::Mat img2D_mat = int2mat(img_2D, inputHeight, inputWidth);//original image
  cv::Mat result_mat = postprocess(paste_mat, img2D_mat, cropAxis[1] + initCropY0 , cropAxis[0] + initCropY0);
  //  cv::Mat result_mat =  paste_mat;
    //cv::addWeighted(result_mat,0.8,img2D_mat,1.0,0,result_mat);
    int ***result2D = new int**[outputHeight];
    for(int i = 0; i< outputHeight; i++){
        result2D[i] = new int*[outputWidth];
        for(int j = 0; j < outputWidth; j++){
            result2D[i][j] = new int[outputChannel];
            result2D[i][j][0] =  result_mat.at<cv::Vec3b>(i,j)[2];
            result2D[i][j][1] =  result_mat.at<cv::Vec3b>(i,j)[1];
            result2D[i][j][2] =  result_mat.at<cv::Vec3b>(i,j)[0];
            //Mat 是 BGR
        }
    }
//這裡開始=======================================================================================================================================================================
   // 裁切完還是2D 所以先轉成1D
   /*int * result = twoDtooneD(result2D, outputHeight, outputWidth);
   // 把單一channel 1D array 轉回24bit
   auto self_rgb_colors = std::make_unique<int[]>(outputPixelNumber);
   for (int i = 0; i < outputPixelNumber; i++){
     self_rgb_colors[i] = (255u & 0xff) << 24 |(result[i] & 0xff) << 16|
             (result[i] & 0xff) << 8|
             (result[i] & 0xff);
   }*/
    int i = 0;
  auto self_rgb_colors = std::make_unique<int[]>(outputPixelNumber);
  for(int h = 0; h < outputHeight; h++){
      for(int w = 0; w < outputWidth; w++){
          int r = result2D[h][w][0];
          int g = result2D[h][w][1];
          int b = result2D[h][w][2];
          self_rgb_colors[i++] = (255u & 0xff) << 24 |(r & 0xff) << 16 |(g & 0xff) << 8|(b & 0xff);
      }
  }

  return self_rgb_colors;
}

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite


//// 這邊底下是原本project的code

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

///*
// * Copyright 2020 The TensorFlow Authors
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     https://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//#include "SuperResolution.h"
//
//#include <android/log.h>
//#include <math.h>
//
//#include <fstream>
//#include <iostream>
//#include <memory>
//#include <string>
//#include <vector>
//
//namespace tflite {
//    namespace examples {
//        namespace superresolution {
//
//// TODO: make it changeable in the UI
//            constexpr int kThreadNum = 4;
//
//            SuperResolution::SuperResolution(const void* model_data, size_t model_size,
//                                             bool use_gpu) {
//              // Load the model
//              model_ = TfLiteModelCreate(model_data, model_size);
//              if (!model_) {
//                LOGE("Failed to create TFLite model");
//                return;
//              }
//
//              // Create the interpreter options
//              options_ = TfLiteInterpreterOptionsCreate();
//
//              // Choose CPU or GPU
//              if (use_gpu) {
//                delegate_ = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
//                TfLiteInterpreterOptionsAddDelegate(options_, delegate_);
//              } else {
//                TfLiteInterpreterOptionsSetNumThreads(options_, kThreadNum);
//              }
//
//              // Create the interpreter
//              interpreter_ = TfLiteInterpreterCreate(model_, options_);
//              if (!interpreter_) {
//                LOGE("Failed to create TFLite interpreter");
//                return;
//              }
//            }
//
//            SuperResolution::~SuperResolution() {
//              // Dispose of the model and interpreter objects
//              if (interpreter_) {
//                TfLiteInterpreterDelete(interpreter_);
//              }
//              if (delegate_) {
//                TfLiteGpuDelegateV2Delete(delegate_);
//              }
//              if (options_) {
//                TfLiteInterpreterOptionsDelete(options_);
//              }
//              if (model_) {
//                TfLiteModelDelete(model_);
//              }
//            }
//
//            bool SuperResolution::IsInterpreterCreated() {
//              if (!interpreter_) {
//                return false;
//              } else {
//                return true;
//              }
//            }
//
//            std::unique_ptr<int[]> SuperResolution::DoSuperResolution(int* lr_img_rgb) {
//              // Allocate tensors and populate the input tensor data
//              TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
//              if (status != kTfLiteOk) {
//                LOGE("Something went wrong when allocating tensors");
//                return nullptr;
//              }
//
//              TfLiteTensor* input_tensor =
//                      TfLiteInterpreterGetInputTensor(interpreter_, 0);
//
//              // Extract RGB values from each pixel
//              float input_buffer[kNumberOfInputPixels * kImageChannels];
//              for (int i = 0, j = 0; i < kNumberOfInputPixels; i++) {
//                // Alpha is ignored
//                input_buffer[j++] = static_cast<float>((lr_img_rgb[i] >> 16) & 0xff);
//                input_buffer[j++] = static_cast<float>((lr_img_rgb[i] >> 8) & 0xff);
//                input_buffer[j++] = static_cast<float>((lr_img_rgb[i]) & 0xff);
//              }
//
//              // Feed input into model
//              status = TfLiteTensorCopyFromBuffer(
//                      input_tensor, input_buffer,
//                      kNumberOfInputPixels * kImageChannels * sizeof(float));
//              if (status != kTfLiteOk) {
//                LOGE("Something went wrong when copying input buffer to input tensor");
//                return nullptr;
//              }
//
//              // Run the interpreter
//              status = TfLiteInterpreterInvoke(interpreter_);
//              if (status != kTfLiteOk) {
//                LOGE("Something went wrong when running the TFLite model");
//                return nullptr;
//              }
//
//              // Extract the output tensor data
//              const TfLiteTensor* output_tensor =
//                      TfLiteInterpreterGetOutputTensor(interpreter_, 0);
//              float output_buffer[kNumberOfOutputPixels * kImageChannels];
//              status = TfLiteTensorCopyToBuffer(
//                      output_tensor, output_buffer,
//                      kNumberOfOutputPixels * kImageChannels * sizeof(float));
//              if (status != kTfLiteOk) {
//                LOGE("Something went wrong when copying output tensor to output buffer");
//                return nullptr;
//              }
//
//              // Postprocess the output from TFLite
//              int clipped_output[kImageChannels];
//              auto rgb_colors = std::make_unique<int[]>(kNumberOfOutputPixels);
//              for (int i = 0; i < kNumberOfOutputPixels; i++) {
//                for (int j = 0; j < kImageChannels; j++) {
//                  clipped_output[j] = std::max<float>(
//                          0, std::min<float>(255, output_buffer[i * kImageChannels + j]));
//                }
//                // When we have RGB values, we pack them into a single pixel.
//                // Alpha is set to 255.
//                rgb_colors[i] = (255u & 0xff) << 24 | (clipped_output[0] & 0xff) << 16 |
//                                (clipped_output[1] & 0xff) << 8 |
//                                (clipped_output[2] & 0xff);
//              }
//
//              return rgb_colors;
//            }
//
//        }  // namespace superresolution
//    }  // namespace examples
//}  // namespace tflite

