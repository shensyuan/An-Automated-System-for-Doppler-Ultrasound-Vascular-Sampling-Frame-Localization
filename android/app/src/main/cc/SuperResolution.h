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

#define LOG_TAG "super_resolution::"
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

namespace tflite {
namespace examples {
namespace superresolution {
const int inputHeight = 700;
const int inputWidth = 900;

const int outputHeight = inputHeight;
const int outputWidth = inputWidth;
const int outputPixelNumber =  outputHeight * outputWidth;

// 可在此設定 range gate 比例
constexpr double kRangeGateRatio = 2.0 / 3.0;

// 備援線束角度：當外部給的取樣線與中心線沒有交點時，改以此角度通過中心線中點重新畫一條線束。
// 角度為影像座標系、相對 +x 軸、順時針為正；75° 即偏離垂直方向 15°。可在此調整。
constexpr double kFallbackBeamAngleDeg = 75.0;

struct PostProcessResult {
    double angle_abs;           // 絕對角度
    double angle_relative;      // 相對角度
    cv::Point center;           // 中心點
    cv::Point intersection_top; // 上交點
    cv::Point intersection_bottom; // 下交點
    cv::Point p_top;            // 線段上端點
    cv::Point p_bottom;         // 線段下端點
    cv::Point2f direction;      // 切線方向
    double vessel_diameter;   // 血管管徑
    cv::Point diameter_top;     // 管徑量測上端點（沿法向掃描到的 mask 邊界）
    cv::Point diameter_bottom;  // 管徑量測下端點
    cv::Point2f perp_direction; // 血管法向（量測方向）
    bool success;               // 是否成功
};

class SuperResolution {
 public:
  SuperResolution(const void* model_data, size_t model_size);
  ~SuperResolution();
  bool IsInterpreterCreated();
  std::unique_ptr<int[]> DoSuperResolution(int* lr_img_rgb, double line_angle, int top_x);
  int doseg(cv::Mat src, int** out,TfLiteInterpreter* interpreter_, bool istop);
  PostProcessResult postprocess(const cv::Mat& mask, const cv::Mat& img_ori, cv::Point p_top, cv::Point p_bottom, int image_h, int image_w);
  cv::Mat apply_clahe(const cv::Mat& gray);
  cv::Mat resize_with_padding(const cv::Mat& img, int size);
  cv::Point get_line_point(int top_x, int top_y, double angle_deg, int length);
  PostProcessResult GetLastResult() const { return last_result_; }
private:
    // 後處理相關函數
    cv::Mat process_single_centerline(const cv::Mat& img_orig, const cv::Mat& mask_224);
    cv::Mat resize_img(const cv::Mat& img, int size);
    cv::Mat crop_img(const cv::Mat& img, int x1, int x2, int y1, int y2);
    void thinningCenterLine(const cv::Mat& src, cv::Mat& dst);
    bool is_valid_centerline(const std::vector<int>& x_pts, const std::vector<int>& y_pts);
    cv::Point2f catmullRomPoint(const cv::Point2f& p0, const cv::Point2f& p1,
                                const cv::Point2f& p2, const cv::Point2f& p3, float t);
    std::vector<cv::Point2f> generateSplinePoints(const std::vector<cv::Point2f>& controlPoints, int totalPoints);

    // Line / Range Gate 函數
    cv::Point find_RangeGate(cv::Point start_pt, cv::Point target_pt, const cv::Mat& img);
    std::pair<cv::Point, cv::Point> get_boundary_intersection_direct(cv::Size mask_shape, cv::Point center_pt, double angle_deg);
    cv::Point2f get_tangent_direction(const cv::Mat& skeleton, cv::Point point, int window_size);
    double calculate_angle_between_vectors(cv::Point p1, cv::Point p2, const cv::Point2f& v_given, bool absolute);
    void draw_tangent(cv::Mat& img, cv::Point point, const cv::Point2f& direction, int length);
    std::pair<cv::Point, cv::Point> draw_perpendicular_line(cv::Mat& image, cv::Point line_p1, cv::Point line_p2,
                                                            cv::Point point, int length, cv::Scalar color, int thickness);
    // 視覺化
    cv::Mat visualizePostProcess(const cv::Mat& img_ori, const PostProcessResult& result);
  PostProcessResult  last_result_;
  TfLiteInterpreter* interpreter_;
  TfLiteModel* model_ = nullptr;
  TfLiteInterpreterOptions* options_ = nullptr;
};

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite
#endif
