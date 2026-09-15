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

#include <android/log.h>
#include <jni.h>

#include <cinttypes>
#include <cstring>
#include <string>

#include "SuperResolution.h"

namespace tflite {
namespace examples {
namespace superresolution {
/**
 * JNI：取得最近一次 DoSuperResolution 的角度結果。
 * @param native_handle initWithByteBufferFromJNI 回傳的 SuperResolution 物件位址
 * @return double[2] = {絕對角 angle_abs, 相對角 angle_relative}（度）；
 *         無有效結果時為 {-1, -1}；handle 為 null 或配置失敗時回傳 nullptr
 */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_PW_1DOPPLER_1ANGLE(
        JNIEnv *env, jobject thiz, jlong native_handle) {

  auto* superRes = reinterpret_cast<SuperResolution *>(native_handle);
  if (!superRes) {
    LOGE("Native handle is null");
    return nullptr;
  }

  PostProcessResult lastResult = superRes->GetLastResult();

  jdoubleArray result = env->NewDoubleArray(2);
  if (result == nullptr) {
    LOGE("Failed to create double array");
    return nullptr;
  }

  jdouble values[2];
  if (lastResult.success) {
    values[0] = lastResult.angle_abs;
    values[1] = lastResult.angle_relative;
    LOGE("Angles: abs=%.2f, rel=%.2f", values[0], values[1]);
  } else {
    values[0] = -1.0;
    values[1] = -1.0;
    LOGE("No valid result available");
  }

  env->SetDoubleArrayRegion(result, 0, 2, values);
  return result;
}

/**
 * JNI：取得中心線與線束交點（或備援中位點）的座標。
 * @param native_handle SuperResolution 物件位址
 * @return int[2] = {x, y}（原圖像素座標）；無有效結果時為 {-1, -1}；失敗回傳 nullptr
 */
extern "C" JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_PW_1Center_1Point(
        JNIEnv *env, jobject thiz, jlong native_handle) {

  auto* superRes = reinterpret_cast<SuperResolution *>(native_handle);
  if (!superRes) {
    LOGE("Native handle is null");
    return nullptr;
  }

  PostProcessResult lastResult = superRes->GetLastResult();

  jintArray result = env->NewIntArray(2);
  if (result == nullptr) {
    LOGE("Failed to create int array");
    return nullptr;
  }

  jint values[2];
  if (lastResult.success) {
    values[0] = lastResult.center.x;
    values[1] = lastResult.center.y;
    LOGE("Center point: (%d, %d)", values[0], values[1]);
  } else {
    values[0] = -1;
    values[1] = -1;
    LOGE("No valid result available");
  }

  env->SetIntArrayRegion(result, 0, 2, values);
  return result;
}

/**
 * JNI：取得 Range Gate 上下位置的 y 座標。
 * 位置為 center 往上下血管邊界方向各前進 kRangeGateRatio 比例處。
 * @param native_handle SuperResolution 物件位址
 * @return int[2] = {top_y, bottom_y}；無有效結果時為 {-1, -1}；失敗回傳 nullptr
 */
extern "C" JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_PW_1RANGE_1GATE(
        JNIEnv *env, jobject thiz, jlong native_handle) {

  auto* superRes = reinterpret_cast<SuperResolution *>(native_handle);
  if (!superRes) {
    LOGE("Native handle is null");
    return nullptr;
  }

  PostProcessResult lastResult = superRes->GetLastResult();

  jintArray result = env->NewIntArray(2);
  if (result == nullptr) {
    LOGE("Failed to create int array");
    return nullptr;
  }

  jint values[2];
  if (lastResult.success) {
    // 新版：回傳管徑 2/3 處的 Y 座標 (往中心線靠近)
    cv::Point range_top = lastResult.center + (lastResult.intersection_top - lastResult.center) * kRangeGateRatio;
    cv::Point range_bottom = lastResult.center + (lastResult.intersection_bottom - lastResult.center) * kRangeGateRatio;
    values[0] = range_top.y;
    values[1] = range_bottom.y;
    LOGE("Range gate Y (2/3): top=%d, bottom=%d", values[0], values[1]);
  } else {
    values[0] = -1;
    values[1] = -1;
    LOGE("No valid result available");
  }

  env->SetIntArrayRegion(result, 0, 2, values);
  return result;
}

/**
 * JNI：取得沿線束方向量到的血管寬度（上下邊界交點的歐氏距離）。
 * @param native_handle SuperResolution 物件位址
 * @return 寬度（像素）；handle 為 null 或無有效結果時 -1.0
 */
extern "C" JNIEXPORT jdouble JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_PW_1VESSEL_1DIAMETER(
        JNIEnv *env, jobject thiz, jlong native_handle) {

  auto* superRes = reinterpret_cast<SuperResolution *>(native_handle);
  if (!superRes) {
    LOGE("Native handle is null");
    return -1.0;
  }

  PostProcessResult lastResult = superRes->GetLastResult();

  if (lastResult.success) {
    double vessel_width = cv::norm(lastResult.intersection_bottom - lastResult.intersection_top);
    LOGE("Vessel diameter: %.2f pixels", vessel_width);
    return vessel_width;
  } else {
    LOGE("No valid result available");
    return -1.0;
  }
}

/**
 * JNI：在原圖上偵測綠色都卜勒取樣線，回傳其角度與頂端 x。
 * 以 BGR 範圍 (30,130,30)~(95,255,255) 取綠色像素，最高點 / 最低點連線求角度。
 * @param image_data ARGB 像素陣列
 * @param width      影像寬
 * @param height     影像高
 * @return double[2] = {角度（度，垂直為 0、右偏為正）, 頂端 x}；找不到綠線時 {9999, -1}
 */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_detectLineFromJNI(
        JNIEnv *env, jobject thiz, jintArray image_data, jint width, jint height) {

  jint *img_array = env->GetIntArrayElements(image_data, nullptr);
  if (img_array == nullptr) {
    return nullptr;
  }

  // 轉換為 cv::Mat
  cv::Mat bgr_img(height, width, CV_8UC3);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int argb = img_array[i * width + j];
      uint8_t r = (argb >> 16) & 0xff;
      uint8_t g = (argb >> 8) & 0xff;
      uint8_t b = argb & 0xff;
      bgr_img.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
    }
  }

  cv::Mat color_img;
  if (bgr_img.channels() == 1) {
    cv::cvtColor(bgr_img, color_img, cv::COLOR_GRAY2BGR);
  } else {
    color_img = bgr_img.clone();
  }

  cv::Scalar lower_green(30, 130, 30);
  cv::Scalar upper_green(95, 255, 255);

  cv::Mat mask;
  cv::inRange(color_img, lower_green, upper_green, mask);

  std::vector<cv::Point> points;
  cv::findNonZero(mask, points);

  jdoubleArray result = env->NewDoubleArray(2);
  jdouble values[2];

  if (points.empty()) {
    values[0] = 9999.0;
    values[1] = -1;
  } else {
    auto p_top = *std::min_element(points.begin(), points.end(),
                                   [](const cv::Point& a, const cv::Point& b) {
                                       return a.y < b.y;
                                   });

    auto p_bottom = *std::max_element(points.begin(), points.end(),
                                      [](const cv::Point& a, const cv::Point& b) {
                                          return a.y < b.y;
                                      });

    double dx = static_cast<double>(p_bottom.x - p_top.x);
    double dy = static_cast<double>(p_bottom.y - p_top.y);

    double alpha_rad = std::atan(std::abs(dx / dy));
    double alpha_deg = alpha_rad * 180.0 / CV_PI;

    if (dx * dy < 0) {
      values[0] = -alpha_deg;
    } else {
      values[0] = alpha_deg;
    }
    values[1] = p_top.x;
  }

  env->SetDoubleArrayRegion(result, 0, 2, values);
  env->ReleaseIntArrayElements(image_data, img_array, JNI_ABORT);

  return result;
}

/**
 * JNI：去除超音波介面上的紅 / 綠色疊圖（取樣線、血流色標等）。
 * HSV 遮罩（綠 H 40~80、紅 H 0~10 與 170~180，S/V ≥ 80）膨脹 2 次後以 Telea inpaint 填補。
 * @param image_data ARGB 像素陣列
 * @param width      影像寬
 * @param height     影像高
 * @return 處理後的 ARGB 像素陣列（同尺寸）
 */
extern "C" JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_removeGreenRedFromJNI(
        JNIEnv *env, jobject thiz, jintArray image_data, jint width, jint height) {

  jint *img_array = env->GetIntArrayElements(image_data, nullptr);
  if (img_array == nullptr) {
    return nullptr;
  }

  // 轉換為 cv::Mat
  cv::Mat bgr_img(height, width, CV_8UC3);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int argb = img_array[i * width + j];
      uint8_t r = (argb >> 16) & 0xff;
      uint8_t g = (argb >> 8) & 0xff;
      uint8_t b = argb & 0xff;
      bgr_img.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
    }
  }

  cv::Mat color_img;
  if (bgr_img.channels() == 1) {
    cv::cvtColor(bgr_img, color_img, cv::COLOR_GRAY2BGR);
  } else {
    color_img = bgr_img.clone();
  }

  cv::Mat hsv;
  cv::cvtColor(color_img, hsv, cv::COLOR_BGR2HSV);

  // Green mask
  cv::Scalar lower_green(40, 80, 80);
  cv::Scalar upper_green(80, 255, 255);
  cv::Mat mask_green;
  cv::inRange(hsv, lower_green, upper_green, mask_green);

  // Red mask (two ranges)
  cv::Scalar lower_red1(0, 80, 80);
  cv::Scalar upper_red1(10, 255, 255);
  cv::Scalar lower_red2(170, 80, 80);
  cv::Scalar upper_red2(180, 255, 255);

  cv::Mat mask_red1, mask_red2;
  cv::inRange(hsv, lower_red1, upper_red1, mask_red1);
  cv::inRange(hsv, lower_red2, upper_red2, mask_red2);

  cv::Mat mask_red;
  cv::bitwise_or(mask_red1, mask_red2, mask_red);

  // Combine masks
  cv::Mat mask;
  cv::bitwise_or(mask_green, mask_red, mask);

  // Dilate
  cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
  cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 2);

  // Inpaint
  cv::Mat result;
  cv::inpaint(color_img, mask, result, 5, cv::INPAINT_TELEA);

  // 轉回 ARGB
  jintArray output = env->NewIntArray(width * height);
  jint output_array[width * height];

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
      uint8_t b = pixel[0];
      uint8_t g = pixel[1];
      uint8_t r = pixel[2];
      output_array[i * width + j] = (255u << 24) | (r << 16) | (g << 8) | b;
    }
  }

  env->SetIntArrayRegion(output, 0, width * height, output_array);
  env->ReleaseIntArrayElements(image_data, img_array, JNI_ABORT);

  return output;
}

/**
 * JNI：執行完整 pipeline（DoSuperResolution）並回傳視覺化影像。
 * @param native_handle SuperResolution 物件位址
 * @param low_res_rgb   inputWidth×inputHeight 的 ARGB 像素陣列（建議先經 removeGreenRedFromJNI）
 * @param line_angle    detectLineFromJNI 得到的取樣線角度
 * @param top_x         detectLineFromJNI 得到的取樣線頂端 x
 * @return 同尺寸 ARGB 視覺化影像；interpreter 未建立或 pipeline 失敗時 nullptr
 */
extern "C" JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_superResolutionFromJNI(
    JNIEnv *env, jobject thiz, jlong native_handle, jintArray low_res_rgb,
    jdouble line_angle, jint top_x) {
  jint *lr_img_rgb = env->GetIntArrayElements(low_res_rgb, NULL);

  auto* superRes = reinterpret_cast<SuperResolution *>(native_handle);
  if (!superRes->IsInterpreterCreated()) {
    env->ReleaseIntArrayElements(low_res_rgb, lr_img_rgb, JNI_ABORT);
    return nullptr;
  }

  auto sr_rgb_colors = superRes->DoSuperResolution(
          static_cast<int *>(lr_img_rgb),
          static_cast<double>(line_angle),
          static_cast<int>(top_x));

  if (!sr_rgb_colors) {
    return nullptr;  // super resolution failed
  }
  jintArray sr_img_rgb = env->NewIntArray(outputPixelNumber);
  env->SetIntArrayRegion(sr_img_rgb, 0, outputPixelNumber,
                         sr_rgb_colors.get());

  // Clean up before we return
  env->ReleaseIntArrayElements(low_res_rgb, lr_img_rgb, JNI_COMMIT);

  return sr_img_rgb;
}

/**
 * JNI：由 direct ByteBuffer 中的 .tflite 建立 SuperResolution 物件。
 * @param model_buffer 映射模型檔的 MappedByteBuffer（需為 direct buffer）
 * @return 物件位址（jlong）；interpreter 建立失敗時 0
 */
extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_initWithByteBufferFromJNI(
    JNIEnv *env, jobject thiz, jobject model_buffer) {
  const void *model_data =
      static_cast<void *>(env->GetDirectBufferAddress(model_buffer));
  jlong model_size_bytes = env->GetDirectBufferCapacity(model_buffer);
  SuperResolution *super_resolution = new SuperResolution(
      model_data, static_cast<size_t>(model_size_bytes));
  if (super_resolution->IsInterpreterCreated()) {
    LOGI("Interpreter is created successfully");
    return reinterpret_cast<jlong>(super_resolution);
  } else {
    delete super_resolution;
    return 0;
  }
}

/**
 * JNI：釋放 initWithByteBufferFromJNI 建立的物件。
 * @param native_handle SuperResolution 物件位址
 */
extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_examples_superresolution_MainActivity_deinitFromJNI(
    JNIEnv *env, jobject thiz, jlong native_handle) {
  delete reinterpret_cast<SuperResolution*>(native_handle);
}

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite
