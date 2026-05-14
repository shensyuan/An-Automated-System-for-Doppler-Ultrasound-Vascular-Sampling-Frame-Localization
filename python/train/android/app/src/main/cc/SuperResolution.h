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

#ifndef NATIVE_LIBS_SUPERRESOLUTION_H
#define NATIVE_LIBS_SUPERRESOLUTION_H
#include <opencv2/opencv.hpp>
#include <string>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#define LOG_TAG "super_resolution::"
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

namespace tflite {
namespace examples {
namespace superresolution {
    //註解部分是5/26會議完後提出的變更
    //input圖片的解析度
const int inputHeight = 886;//562
const int inputWidth = 720;

const int inputPixelNumber = inputHeight * inputWidth;
const int initCropX0 = 40;//0
const int initCropX1 = 650;//inputWidth
const int initCropY0 = 200;//0
const int initCropY1 = 600;//inputHeight;
const int initCropHeight = initCropY1-initCropY0;
const int initCropWidth = initCropX1-initCropX0;

const int modelinputHeight = 128;
const int modelinputWidth = 512;
const int modelinputChannels = 2;
//output圖片的解析度
const int modeloutputHeight = 128;
const int modeloutputWidth = 512;
const int modeloutputChannel = 1;
const int outputthreshold = 5;// predict 出來的圖是灰的 所以會做一個threshold filtering

const int outputHeight = inputHeight;
const int outputWidth = inputWidth;
const int outputChannel = 3;
const int outputPixelNumber =  outputHeight * outputWidth;
class SuperResolution {
 public:
  SuperResolution(const void* model_data, size_t model_size, bool use_gpu);
  ~SuperResolution();
  bool IsInterpreterCreated();
  // DoSuperResolution() performs super resolution on a low resolution image. It
  // returns a valid pointer if successful and nullptr if unsuccessful.
  // lr_img_rgb: the pointer to the RGB array extracted from low resolution
  // image
  std::unique_ptr<int[]> DoSuperResolution(int* lr_img_rgb);
  int** mat2int(cv::Mat src);
  cv::Mat int2mat(int** src, int rows, int cols);
  int ** oneDtotwoD(int * img_1D, int height, int width);
  int * twoDtooneD(int ** img_2D, int height, int width);
  int ** InitCrop(int ** img, int cropX0,int cropX1, int cropY0, int cropY1);
  int * get_cropImg_axis(int ** img, int height, int width);
  float * smooth(float* sum, int size, int k);
  int ** pasteBack(int** top, int** bot, int cropUp, int axis, int cropDown, int oriHeight, int oriWidth);
  cv::Mat eliNoise(cv::Mat src, int dark, int percentage);
  cv::Mat imgsToPrewitt(cv::Mat srcImage);
  int doseg(cv::Mat src, int** out,TfLiteInterpreter* interpreter_, bool istop);
  cv::Mat postprocess(cv::Mat src, cv::Mat osrc,int croplinemid , int croplineup);
    private:
  // TODO: use unique_ptr
  TfLiteInterpreter* interpreter_;
  TfLiteModel* model_ = nullptr;
  TfLiteInterpreterOptions* options_ = nullptr;
  TfLiteDelegate* delegate_ = nullptr;
};

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite
#endif
