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
constexpr int kThreadNum = 4;

#define CV_PI 3.14159265358979323846
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "SuperResolution", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "SuperResolution", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SuperResolution", __VA_ARGS__)

// ==================== 前處理 ====================
/**
 * 對灰階影像套用 CLAHE（clipLimit=2.0, tile 8×8）做局部對比強化。
 * @param gray 8-bit 單通道灰階影像
 * @return 強化後的灰階影像，尺寸與輸入相同
 */
cv::Mat SuperResolution::apply_clahe(const cv::Mat& gray) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    return enhanced;
}


/**
 * 等比縮放影像使長邊等於 size，短邊兩側補黑邊，輸出 size×size 的正方形影像。
 * @param img  任意尺寸的輸入影像（通道數不限）
 * @param size 目標邊長（本專案為 224）
 * @return size×size 的補邊影像
 */
cv::Mat SuperResolution::resize_with_padding(const cv::Mat& img, int size) {
    int h = img.rows;
    int w = img.cols;

    double scale = static_cast<double>(size) / static_cast<double>(std::max(h, w));

    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    int pad_top = (size - new_h) / 2;
    int pad_bottom = size - new_h - pad_top;
    int pad_left = (size - new_w) / 2;
    int pad_right = size - new_w - pad_left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(0));

    return padded;
}


/**
 * 由線段頂端點、角度與長度，計算線段另一端（下端）的座標。
 * 角度定義：垂直向下為 0°，往右偏為正（dx = sin, dy = cos）。
 * @param top_x     頂端點 x
 * @param top_y     頂端點 y
 * @param angle_deg 線段相對垂直方向的角度（度）
 * @param length    線段長度（像素）
 * @return 下端點座標（四捨五入為整數）
 */
cv::Point SuperResolution::get_line_point(int top_x, int top_y, double angle_deg, int length) {
    double angle_rad = angle_deg * CV_PI / 180.0;
    double dx = std::sin(angle_rad) * length;
    double dy = std::cos(angle_rad) * length;

    int bottom_x = static_cast<int>(std::round(top_x + dx));
    int bottom_y = static_cast<int>(std::round(top_y + dy));

    return cv::Point(bottom_x, bottom_y);
}

// ==================== 後處理 ====================
/**
 * 等比縮放影像使「短邊」等於 size（長邊依比例放大，不補邊、不裁切）。
 * @param img  輸入影像
 * @param size 目標短邊長度
 * @return 縮放後影像
 */
cv::Mat SuperResolution::resize_img(const cv::Mat& img, int size) {
    int h = img.rows;
    int w = img.cols;
    double scale = static_cast<double>(size) / std::min(h, w);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    return resized;
}

/**
 * 裁切影像的矩形區域 [x1, x2) × [y1, y2)。
 * @param img 輸入影像
 * @param x1  左邊界（含）
 * @param x2  右邊界（不含）
 * @param y1  上邊界（含）
 * @param y2  下邊界（不含）
 * @return 裁切後的獨立副本（clone）
 */
cv::Mat SuperResolution::crop_img(const cv::Mat& img, int x1, int x2, int y1, int y2) {
    return img(cv::Range(y1, y2), cv::Range(x1, x2)).clone();
}

/**
 * Zhang-Suen 細線化：把二值 mask 迭代削薄成單像素寬的骨架。
 * @param src 輸入 mask（>127 視為前景；若為彩色會先轉灰階）
 * @param dst 輸出骨架（CV_8UC1，前景 255 / 背景 0），與 src 同尺寸
 */
void SuperResolution::thinningCenterLine(const cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    if (dst.channels() > 1) {
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(dst, dst, 127, 255, cv::THRESH_BINARY);

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;
    std::vector<cv::Point> toDelete;

    do {
        // Step 1
        toDelete.clear();
        for (int i = 1; i < dst.rows - 1; i++) {
            for (int j = 1; j < dst.cols - 1; j++) {
                if (dst.at<uchar>(i, j) == 0) continue;

                uchar p2 = dst.at<uchar>(i, j + 1) / 255;
                uchar p3 = dst.at<uchar>(i + 1, j + 1) / 255;
                uchar p4 = dst.at<uchar>(i + 1, j) / 255;
                uchar p5 = dst.at<uchar>(i + 1, j - 1) / 255;
                uchar p6 = dst.at<uchar>(i, j - 1) / 255;
                uchar p7 = dst.at<uchar>(i - 1, j - 1) / 255;
                uchar p8 = dst.at<uchar>(i - 1, j) / 255;
                uchar p9 = dst.at<uchar>(i - 1, j + 1) / 255;

                int P1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int S = 0;
                if (p2 == 0 && p3 == 1) S++;
                if (p3 == 0 && p4 == 1) S++;
                if (p4 == 0 && p5 == 1) S++;
                if (p5 == 0 && p6 == 1) S++;
                if (p6 == 0 && p7 == 1) S++;
                if (p7 == 0 && p8 == 1) S++;
                if (p8 == 0 && p9 == 1) S++;
                if (p9 == 0 && p2 == 1) S++;

                if (P1 >= 2 && P1 <= 6 && S == 1 &&
                    p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0) {
                    toDelete.push_back(cv::Point(j, i));
                }
            }
        }
        for (const auto& pt : toDelete) {
            dst.at<uchar>(pt.y, pt.x) = 0;
        }

        // Step 2
        toDelete.clear();
        for (int i = 1; i < dst.rows - 1; i++) {
            for (int j = 1; j < dst.cols - 1; j++) {
                if (dst.at<uchar>(i, j) == 0) continue;

                uchar p2 = dst.at<uchar>(i, j + 1) / 255;
                uchar p3 = dst.at<uchar>(i + 1, j + 1) / 255;
                uchar p4 = dst.at<uchar>(i + 1, j) / 255;
                uchar p5 = dst.at<uchar>(i + 1, j - 1) / 255;
                uchar p6 = dst.at<uchar>(i, j - 1) / 255;
                uchar p7 = dst.at<uchar>(i - 1, j - 1) / 255;
                uchar p8 = dst.at<uchar>(i - 1, j) / 255;
                uchar p9 = dst.at<uchar>(i - 1, j + 1) / 255;

                int P1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int S = 0;
                if (p2 == 0 && p3 == 1) S++;
                if (p3 == 0 && p4 == 1) S++;
                if (p4 == 0 && p5 == 1) S++;
                if (p5 == 0 && p6 == 1) S++;
                if (p6 == 0 && p7 == 1) S++;
                if (p7 == 0 && p8 == 1) S++;
                if (p8 == 0 && p9 == 1) S++;
                if (p9 == 0 && p2 == 1) S++;

                if (P1 >= 2 && P1 <= 6 && S == 1 &&
                    p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) {
                    toDelete.push_back(cv::Point(j, i));
                }
            }
        }
        for (const auto& pt : toDelete) {
            dst.at<uchar>(pt.y, pt.x) = 0;
        }

        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);
}

/**
 * 判斷一條候選中心線是否合理，過濾掉太短、太窄或上下抖動過大的片段。
 * 條件：點數 ≥ 15、x 方向跨度 ≥ 20 px、相鄰點 |Δy| 總和 / 寬度 ≤ 1.5。
 * @param x_pts 依 x 排序的中心線各點 x 座標
 * @param y_pts 對應的 y 座標
 * @return 通過檢查回傳 true
 */
bool SuperResolution::is_valid_centerline(const std::vector<int>& x_pts, const std::vector<int>& y_pts) {
    if (static_cast<int>(x_pts.size()) < 15) return false;

    int min_x = *std::min_element(x_pts.begin(), x_pts.end());
    int max_x = *std::max_element(x_pts.begin(), x_pts.end());
    int width = max_x - min_x;

    double y_variability = 0.0;
    for (size_t i = 1; i < y_pts.size(); i++) {
        y_variability += std::abs(y_pts[i] - y_pts[i - 1]);
    }
    y_variability /= (width + 1e-6);

    if (width < 20) return false;
    if (y_variability > 1.5) return false;

    return true;
}

/**
 * 計算 Catmull-Rom 樣條在四個控制點 p0~p3 之間、參數 t 處的位置（曲線段落在 p1~p2 之間）。
 * @param p0 前一控制點
 * @param p1 段起點
 * @param p2 段終點
 * @param p3 後一控制點
 * @param t  段內參數，0~1
 * @return 曲線上的點
 */
cv::Point2f SuperResolution::catmullRomPoint(const cv::Point2f& p0, const cv::Point2f& p1,
                                             const cv::Point2f& p2, const cv::Point2f& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    float x = 0.5f * ((2.0f * p1.x) +
                      (-p0.x + p2.x) * t +
                      (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * t2 +
                      (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t3);
    float y = 0.5f * ((2.0f * p1.y) +
                      (-p0.y + p2.y) * t +
                      (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
                      (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3);

    return cv::Point2f(x, y);
}

/**
 * 用 Catmull-Rom 樣條把一串控制點內插成平滑曲線上的密集取樣點。
 * 控制點只有 2 個時退化為線性內插；首尾段以外推的虛擬控制點補齊。
 * @param controlPoints 依序排列的控制點（至少 2 個，否則回傳空陣列）
 * @param totalPoints   輸出點總數（平均分配到各段）
 * @return 曲線上的取樣點
 */
std::vector<cv::Point2f> SuperResolution::generateSplinePoints(const std::vector<cv::Point2f>& controlPoints, int totalPoints) {
    std::vector<cv::Point2f> result;
    int n = static_cast<int>(controlPoints.size());
    if (n < 2) return result;

    if (n == 2) {
        for (int i = 0; i < totalPoints; i++) {
            float t = static_cast<float>(i) / (totalPoints - 1);
            float x = controlPoints[0].x + t * (controlPoints[1].x - controlPoints[0].x);
            float y = controlPoints[0].y + t * (controlPoints[1].y - controlPoints[0].y);
            result.push_back(cv::Point2f(x, y));
        }
        return result;
    }

    int segments = n - 1;
    int pointsPerSegment = totalPoints / segments;
    int extra = totalPoints - pointsPerSegment * segments;

    for (int seg = 0; seg < segments; seg++) {
        cv::Point2f p0, p1, p2, p3;
        p1 = controlPoints[seg];
        p2 = controlPoints[seg + 1];

        if (seg == 0) {
            p0 = p1 - (p2 - p1);
        } else {
            p0 = controlPoints[seg - 1];
        }

        if (seg + 2 < n) {
            p3 = controlPoints[seg + 2];
        } else {
            p3 = p2 + (p2 - p1);
        }

        int nPts = pointsPerSegment + (seg < extra ? 1 : 0);
        for (int j = 0; j < nPts; j++) {
            float t = static_cast<float>(j) / nPts;
            result.push_back(catmullRomPoint(p0, p1, p2, p3, t));
        }
    }

    return result;
}

/**
 * 從 224×224 的分割 mask 擷取血管中心線，並放大到原圖尺寸。
 * 流程：二值化 → 8 連通元件（<20 px 略過）→ 距離轉換 → Zhang-Suen 骨架 →
 * 每個 x 欄取距離值最大的骨架點 → is_valid_centerline 篩選 → 座標縮放到原圖 →
 * Catmull-Rom 樣條內插 2000 點畫進輸出 mask。
 * @param img_orig 原圖（只用其尺寸決定輸出大小與縮放比例）
 * @param mask_224 模型輸出的 224×224 mask（CV_8UC1）
 * @return 與 img_orig 同尺寸的中心線 mask（CV_8UC1，中心線為 255）
 */
cv::Mat SuperResolution::process_single_centerline(const cv::Mat& img_orig, const cv::Mat& mask_224) {
    int h_orig = img_orig.rows;
    int w_orig = img_orig.cols;
    double scale_x = static_cast<double>(w_orig) / 224.0;
    double scale_y = static_cast<double>(h_orig) / 224.0;

    cv::Mat centerline_mask = cv::Mat::zeros(h_orig, w_orig, CV_8UC1);

    cv::Mat binary_mask;
    cv::threshold(mask_224, binary_mask, 127, 255, cv::THRESH_BINARY);

    cv::Mat labeled_array;
    int num_features = cv::connectedComponents(binary_mask, labeled_array, 8, CV_32S);

    for (int i = 1; i <= num_features; i++) {
        cv::Mat single_region = (labeled_array == i);
        if (cv::countNonZero(single_region) < 20) continue;

        cv::Mat dist_map;
        cv::distanceTransform(single_region, dist_map, cv::DIST_L2, cv::DIST_MASK_PRECISE);

        cv::Mat skel_region;
        thinningCenterLine(single_region, skel_region);

        std::vector<cv::Point> skel_points;
        cv::findNonZero(skel_region, skel_points);

        std::map<int, std::pair<int, float>> region_best_pts;
        for (const auto& pt : skel_points) {
            int x = pt.x;
            int y = pt.y;
            float val = dist_map.at<float>(y, x);
            auto it = region_best_pts.find(x);
            if (it == region_best_pts.end() || val > it->second.second) {
                region_best_pts[x] = {y, val};
            }
        }

        std::vector<int> sorted_x, sorted_y;
        for (const auto& kv : region_best_pts) {
            sorted_x.push_back(kv.first);
            sorted_y.push_back(kv.second.first);
        }

        if (!is_valid_centerline(sorted_x, sorted_y)) continue;

        std::vector<cv::Point2f> pts_scaled;
        for (size_t j = 0; j < sorted_x.size(); j++) {
            pts_scaled.push_back(cv::Point2f(
                    sorted_x[j] * scale_x,
                    sorted_y[j] * scale_y
            ));
        }

        std::vector<cv::Point2f> fine_pts = generateSplinePoints(pts_scaled, 2000);

        for (const auto& pt : fine_pts) {
            int ix = static_cast<int>(std::round(pt.x));
            int iy = static_cast<int>(std::round(pt.y));
            if (ix >= 0 && ix < w_orig && iy >= 0 && iy < h_orig) {
                centerline_mask.at<uchar>(iy, ix) = 255;
            }
        }
    }

    return centerline_mask;
}

/**
 * 從 start_pt 沿直線往 target_pt 逐像素前進，找到第一個離開血管 mask 的位置（血管邊界）。
 * 碰到 mask 為 0 的像素時回退 2 px 並 clamp 到影像範圍內作為結果；
 * 若一路都在 mask 內或走出影像，回傳最後一個有效點。
 * @param start_pt  起點（通常是中心線與線束的交點）
 * @param target_pt 目標方向上的點（線束與 mask 的最上 / 最下交點）
 * @param img       血管 mask（CV_8UC1）
 * @return 血管邊界點座標
 */
cv::Point SuperResolution::find_RangeGate(cv::Point start_pt, cv::Point target_pt, const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    double start_x = static_cast<double>(start_pt.x);
    double start_y = static_cast<double>(start_pt.y);
    double target_x = static_cast<double>(target_pt.x);
    double target_y = static_cast<double>(target_pt.y);

    double dx = target_x - start_x;
    double dy = target_y - start_y;
    double distance = std::sqrt(dx * dx + dy * dy);

    if (distance == 0) return start_pt;

    double ux = dx / distance;
    double uy = dy / distance;

    cv::Point last_pt = start_pt;
    for (double d = 0.0; d < distance; d += 1.0) {
        int curr_x = static_cast<int>(start_x + d * ux);
        int curr_y = static_cast<int>(start_y + d * uy);

        if (!(curr_y >= 0 && curr_y < h && curr_x >= 0 && curr_x < w)) {
            return last_pt;
        }

        if (img.at<uchar>(curr_y, curr_x) == 0) {
            int final_x = static_cast<int>(std::round(curr_x - ux * 2));
            int final_y = static_cast<int>(std::round(curr_y - uy * 2));
            final_x = std::max(0, std::min(w - 1, final_x));
            final_y = std::max(0, std::min(h - 1, final_y));
            return cv::Point(final_x, final_y);
        }

        last_pt = cv::Point(curr_x, curr_y);
    }

    return last_pt;
}

/**
 * 通過 center_pt、以 angle_deg 為方向的直線，與影像四邊的兩個交點。
 * 角度定義：相對 +x 軸、順時針為正（dx = cos, dy = sin）。
 * @param mask_shape 影像尺寸
 * @param center_pt  直線通過的點
 * @param angle_deg  直線方向角（度）
 * @return {上端點, 下端點}（依 y 座標排序）
 */
std::pair<cv::Point, cv::Point> SuperResolution::get_boundary_intersection_direct(cv::Size mask_shape, cv::Point center_pt, double angle_deg) {
    int h = mask_shape.height;
    int w = mask_shape.width;
    double cx = static_cast<double>(center_pt.x);
    double cy = static_cast<double>(center_pt.y);

    double rad = angle_deg * CV_PI / 180.0;
    double dx = std::cos(rad);
    double dy = std::sin(rad);

    std::vector<double> distances;
    double epsilon = 1e-9;

    if (std::abs(dx) > epsilon) {
        distances.push_back((0.0 - cx) / dx);
        distances.push_back((w - 1.0 - cx) / dx);
    }
    if (std::abs(dy) > epsilon) {
        distances.push_back((0.0 - cy) / dy);
        distances.push_back((h - 1.0 - cy) / dy);
    }

    double t_pos = 1e18;
    double t_neg = -1e18;
    for (double t : distances) {
        if (t > 0 && t < t_pos) t_pos = t;
        if (t < 0 && t > t_neg) t_neg = t;
    }

    cv::Point pt_edge_1(
            static_cast<int>(cx + t_pos * dx),
            static_cast<int>(cy + t_pos * dy)
    );
    cv::Point pt_edge_2(
            static_cast<int>(cx + t_neg * dx),
            static_cast<int>(cy + t_neg * dy)
    );

    cv::Point p_top, p_bottom;
    if (pt_edge_1.y > pt_edge_2.y) {
        p_bottom = pt_edge_1;
        p_top = pt_edge_2;
    } else {
        p_bottom = pt_edge_2;
        p_top = pt_edge_1;
    }

    return {p_top, p_bottom};
}

/**
 * 以 PCA（SVD）估計骨架上某點的切線方向：取 point 周圍 ±window_size 的骨架點，
 * 對其座標去均值後做 SVD，第一主成分即切線方向。方向統一翻轉成 x ≤ 0 並正規化。
 * @param skeleton    中心線 mask（CV_8UC1）
 * @param point       要估計切線的點（需在骨架上）
 * @param window_size 取樣視窗半徑（像素）
 * @return 單位切線向量；point 不在骨架上、越界或視窗內點數 < 2 時回傳 (1, 0)
 */
cv::Point2f SuperResolution::get_tangent_direction(const cv::Mat& skeleton, cv::Point point, int window_size) {
    int x0 = point.x;
    int y0 = point.y;
    int h = skeleton.rows;
    int w = skeleton.cols;

    if (skeleton.empty()) {
        return cv::Point2f(1.0f, 0.0f);
    }

    if (!(x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)) {
        return cv::Point2f(1.0f, 0.0f);
    }

    if (skeleton.at<uchar>(y0, x0) == 0) {
        return cv::Point2f(1.0f, 0.0f);
    }

    int x1 = std::max(0, x0 - window_size);
    int x2 = std::min(w, x0 + window_size);
    int y1 = std::max(0, y0 - window_size);
    int y2 = std::min(h, y0 + window_size);

    cv::Mat roi = skeleton(cv::Range(y1, y2), cv::Range(x1, x2));

    std::vector<cv::Point> roi_points;
    cv::findNonZero(roi, roi_points);

    if (roi_points.size() < 2) {
        return cv::Point2f(1.0f, 0.0f);
    }

    cv::Mat points_mat(static_cast<int>(roi_points.size()), 2, CV_32F);
    for (size_t i = 0; i < roi_points.size(); i++) {
        points_mat.at<float>(static_cast<int>(i), 0) = static_cast<float>(roi_points[i].x + x1);
        points_mat.at<float>(static_cast<int>(i), 1) = static_cast<float>(roi_points[i].y + y1);
    }

    cv::Mat mean;
    cv::reduce(points_mat, mean, 0, cv::REDUCE_AVG);
    cv::Mat centered = points_mat - cv::repeat(mean, points_mat.rows, 1);

    if (cv::norm(centered, cv::NORM_L2) < 1e-6f) {
        return cv::Point2f(1.0f, 0.0f);
    }

    cv::Mat svd_w, u, vt;
    cv::Mat float_src;
    centered.convertTo(float_src, CV_32F);
    cv::SVD::compute(float_src, svd_w, u, vt, cv::SVD::FULL_UV);

    cv::Point2f direction(vt.at<float>(0, 0), vt.at<float>(0, 1));

    if (direction.x > 0) {
        direction = -direction;
    }

    float norm = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    if (norm > 0) {
        direction.x /= norm;
        direction.y /= norm;
    } else {
        direction = cv::Point2f(1.0f, 0.0f);
    }

    return direction;
}

/**
 * 計算線束向量 (p1→p2) 與切線向量 v 的夾角。
 * 線束角歸約到 0~180°、切線角歸約到 90~270°（y 軸向上為正的座標系）後相減。
 * @param p1       線束起點
 * @param p2       線束終點
 * @param v        切線方向向量
 * @param absolute false：回傳 0~90° 的銳角；true：回傳歸約後的有號差值（切線角 − 線束角）
 * @return 夾角（度）
 */
double SuperResolution::calculate_angle_between_vectors(cv::Point p1, cv::Point p2, const cv::Point2f& v, bool absolute) {
    cv::Point2f vec_a(
            static_cast<float>(p2.x - p1.x),
            static_cast<float>(p2.y - p1.y)
    );
    cv::Point2f vec_b = v;

    LOGD("=== calculate_angle_between_vectors START ===");
    LOGD("p1: (%d, %d); p2: (%d, %d)", p1.x, p1.y, p2.x, p2.y);

    // 計算原始角度（-180° 到 180°）
    double angle_a_raw = std::atan2(-vec_a.y, vec_a.x) * 180.0 / M_PI;
    double angle_b_raw = std::atan2(-vec_b.y, vec_b.x) * 180.0 / M_PI;
    LOGD("raw angle(beam, direction): %.2f, %.2f", angle_a_raw, angle_b_raw);

    // vector1 歸約到 0~180°
    double angle_a = std::fmod(angle_a_raw, 180.0);
    if (angle_a < 0) angle_a += 180.0;
    if (angle_a >= 180.0) angle_a -= 180.0;

    // vector2 歸約到 90~270°
    double angle_b = std::fmod(angle_b_raw, 360.0);
    if (angle_b < 0) angle_b += 360.0;

    // 映射到 [90, 270)
    if (angle_b < 90.0) {
        angle_b = angle_b + 180.0;  // 0-90° -> 180-270°
    } else if(angle_b > 270.0) {
        angle_b = angle_b - 180.0;  // 270-360° -> 90-180°
    }

    LOGD("norm angle(beam, direction): %.2f, %.2f", angle_a, angle_b);
    // 計算 vector2 - vector1
    double result = angle_b - angle_a;

    if (!absolute) {
        // 確保結果在 0~180° 之間
        if (result < 0) result += 180.0;
        if (result > 180.0) result -= 180.0;

        if (result > 90.0) result = 180.0 - result;

        return result;
    } else {
        // 返回原始差值（可能為負）
        return result;
    }
}

/**
 * 在影像上以 point 為中心、沿 direction 兩側各 length 像素畫一條紅色切線（粗 2）。
 * @param img       目標影像（BGR，就地繪製）
 * @param point     切線中心點
 * @param direction 切線方向（會先正規化）
 * @param length    單側長度（像素）
 */
void SuperResolution::draw_tangent(cv::Mat& img, cv::Point point, const cv::Point2f& direction, int length) {
    double dx = static_cast<double>(direction.x);
    double dy = static_cast<double>(direction.y);
    double norm = std::sqrt(dx * dx + dy * dy);
    dx /= norm;
    dy /= norm;

    cv::Point p1(
            static_cast<int>(point.x - dx * length),
            static_cast<int>(point.y - dy * length)
    );
    cv::Point p2(
            static_cast<int>(point.x + dx * length),
            static_cast<int>(point.y + dy * length)
    );

    cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2);
}

/**
 * 在 point 處畫一條垂直於線段 (line_p1→line_p2) 的短線，總長 length，用來標示 Range Gate。
 * @param image     目標影像（就地繪製）
 * @param line_p1   參考線段起點
 * @param line_p2   參考線段終點
 * @param point     短線中心點
 * @param length    短線總長（像素）
 * @param color     顏色
 * @param thickness 線寬
 * @return 短線兩端點 {p1, p2}；參考線段長度為 0 時回傳 {(-1,-1), (-1,-1)} 且不繪製
 */
std::pair<cv::Point, cv::Point> SuperResolution::draw_perpendicular_line(cv::Mat& image, cv::Point line_p1, cv::Point line_p2,
                                                                         cv::Point point, int length, cv::Scalar color, int thickness) {
    double dx = static_cast<double>(line_p2.x - line_p1.x);
    double dy = static_cast<double>(line_p2.y - line_p1.y);
    double line_length = std::sqrt(dx * dx + dy * dy);

    if (line_length == 0) {
        return {cv::Point(-1, -1), cv::Point(-1, -1)};
    }

    double ux = dx / line_length;
    double uy = dy / line_length;

    double perp_x = -uy;
    double perp_y = ux;

    double half_length = length / 2.0;
    cv::Point p1(
            static_cast<int>(point.x + perp_x * half_length),
            static_cast<int>(point.y + perp_y * half_length)
    );
    cv::Point p2(
            static_cast<int>(point.x - perp_x * half_length),
            static_cast<int>(point.y - perp_y * half_length)
    );

    cv::line(image, p1, p2, color, thickness);
    return {p1, p2};
}


/**
 * 建構子：由記憶體中的 .tflite 模型建立 TFLite model、options（kThreadNum 執行緒）與 interpreter。
 * 任一步失敗會記錄錯誤並提早返回，之後可用 IsInterpreterCreated() 檢查。
 * @param model_data 模型檔內容的起始位址（需在物件存活期間保持有效）
 * @param model_size 模型大小（bytes）
 */
SuperResolution::SuperResolution(const void* model_data, size_t model_size) {
  // Load the model
  model_ = TfLiteModelCreate(model_data, model_size);
  if (!model_) {
    LOGE("Failed to create TFLite model");
    return;
  }
  LOGD("Model created successfully");

  // Create the interpreter options
  options_ = TfLiteInterpreterOptionsCreate();

  // Choose CPU or GPU
  TfLiteInterpreterOptionsSetNumThreads(options_, kThreadNum);

  // Create the interpreter
  interpreter_ = TfLiteInterpreterCreate(model_, options_);
  if (!interpreter_) {
    LOGE("Failed to create TFLite interpreter");
    return;
  }

  LOGD("Number of input tensors: %d", TfLiteInterpreterGetInputTensorCount(interpreter_));
  LOGD("Number of output tensors: %d", TfLiteInterpreterGetOutputTensorCount(interpreter_));
}

/**
 * 解構子：依序釋放 interpreter、options 與 model。
 */
SuperResolution::~SuperResolution() {
  // Dispose of the model and interpreter objects
  if (interpreter_) {
    TfLiteInterpreterDelete(interpreter_);
  }
  if (options_) {
    TfLiteInterpreterOptionsDelete(options_);
  }
  if (model_) {
    TfLiteModelDelete(model_);
  }
}

/**
 * @return interpreter 是否成功建立
 */
bool SuperResolution::IsInterpreterCreated() {
  if (!interpreter_) {
    return false;
  } else {
    return true;
  }
}

// ==================== 模型 ====================
/**
 * 執行 TFLite 分割模型，輸出 224×224 的二值 mask。
 * 輸入影像縮放到 224×224、正規化為 (x / 127.5 − 1)，依模型輸入通道數填成 1 或 3 通道；
 * 輸出取第一通道，val × 255 > 128 者設為 255，其餘為 0。
 * @param src          單通道灰階影像（不是 224×224 會自動 resize）
 * @param out          已配置好的 int[224][224] 輸出陣列
 * @param interpreter_ 要使用的 TFLite interpreter
 * @param istop        true 時先將影像上下翻轉
 * @return 成功 0；tensor 取得 / 複製 / 推論失敗或通道數不支援時 -1
 */
int SuperResolution::doseg(cv::Mat src, int** out,TfLiteInterpreter* interpreter_ , bool istop){
    LOGD("=== doseg START ===");

    if (istop) {
        cv::flip(src, src, 0);
        LOGD("Applied flip");
    }


    cv::Mat input_img;
    if (src.rows != 224 || src.cols != 224) {
        cv::resize(src, input_img, cv::Size(224, 224));
        LOGD("Resized input to 224x224");
    } else {
        input_img = src;
        LOGD("Input already 224x224");
    }

    // 檢查輸入的統計值
    cv::Scalar mean_src, stddev_src;
    cv::meanStdDev(input_img, mean_src, stddev_src);
    LOGD("doseg input - mean: %.2f, std: %.2f", mean_src[0], stddev_src[0]);

    // 獲取輸入 tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    if (input_tensor == nullptr) {
        LOGE("Input tensor is null!");
        return -1;
    }

    // 獲取輸入維度
    int input_dims_num = TfLiteTensorNumDims(input_tensor);
    LOGD("Input tensor dimensions count: %d", input_dims_num);

    if (input_dims_num >= 4) {
        int batch = TfLiteTensorDim(input_tensor, 0);
        int height = TfLiteTensorDim(input_tensor, 1);
        int width = TfLiteTensorDim(input_tensor, 2);
        int channels = TfLiteTensorDim(input_tensor, 3);
        LOGD("Model input dims: batch=%d, height=%d, width=%d, channels=%d",
             batch, height, width, channels);
    }

    // 獲取輸入類型
    TfLiteType input_type = TfLiteTensorType(input_tensor);
    LOGD("Input tensor type: %d (kTfLiteFloat32 = 1)", input_type);

    // 決定模型輸入尺寸
    int modelHeight = 224;
    int modelWidth = 224;
    int modelChannels = 3;

    // 從 tensor 獲取實際尺寸
    if (input_dims_num >= 4) {
        modelHeight = TfLiteTensorDim(input_tensor, 1);
        modelWidth = TfLiteTensorDim(input_tensor, 2);
        modelChannels = TfLiteTensorDim(input_tensor, 3);
        LOGD("Using actual model input dims: %dx%dx%d", modelHeight, modelWidth, modelChannels);
    }


    int bufferSize = modelHeight * modelWidth * modelChannels;
    float* input_buffer = new float[bufferSize];

    // 根據輸入通道數填充數據
    if (modelChannels == 1) {
        // 單通道 (灰階)
        for (int i = 0; i < modelHeight; i++) {
            for (int j = 0; j < modelWidth; j++) {
                input_buffer[i * modelWidth + j] = input_img.at<uchar>(i, j) / 127.5f - 1.0f;
            }
        }
    } else if (modelChannels == 3) {
        // 三通道 (RGB) - 將灰階轉成 RGB
        for (int i = 0; i < modelHeight; i++) {
            for (int j = 0; j < modelWidth; j++) {
                float val = input_img.at<uchar>(i, j) / 127.5f - 1.0f;
                int idx = (i * modelWidth + j) * 3;
                input_buffer[idx] = val;     // R
                input_buffer[idx + 1] = val; // G
                input_buffer[idx + 2] = val; // B
            }
        }
    } else {
        LOGE("Unsupported input channels: %d", modelChannels);
        delete[] input_buffer;
        return -1;
    }

    LOGD("input_buffer[0]: %f", input_buffer[0]);

    // 給模型
    TfLiteStatus status = TfLiteTensorCopyFromBuffer(
            input_tensor, input_buffer,
            bufferSize * sizeof(float));
    if (status != kTfLiteOk) {
        LOGE("Failed to copy input buffer to input tensor");
        delete[] input_buffer;
        return -1;
    }
    LOGD("Input copied successfully");

    // 執行模型
    status = TfLiteInterpreterInvoke(interpreter_);
    if (status != kTfLiteOk) {
        LOGE("Failed to run TFLite model, status: %d", status);
        delete[] input_buffer;
        return -1;
    }
    LOGD("Model invoked successfully");

    // 獲取輸出 tensor
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    if (output_tensor == nullptr) {
        LOGE("Output tensor is null!");
        delete[] input_buffer;
        return -1;
    }

    // 獲取輸出維度
    int output_dims_num = TfLiteTensorNumDims(output_tensor);
    LOGD("Output tensor dimensions count: %d", output_dims_num);

    TfLiteQuantizationParams output_quant = TfLiteTensorQuantizationParams(output_tensor);
    LOGD("Output quantization: scale=%f, zero_point=%d", output_quant.scale, output_quant.zero_point);

    // 檢查輸出類型
    TfLiteType output_type = TfLiteTensorType(output_tensor);
    LOGD("Output tensor type: %d (1=float32, 3=int8, 9=uint8)", output_type);

    int outputHeight = 224;
    int outputWidth = 224;
    int outputChannels = 1;

    if (output_dims_num >= 4) {
        outputHeight = TfLiteTensorDim(output_tensor, 1);
        outputWidth = TfLiteTensorDim(output_tensor, 2);
        outputChannels = TfLiteTensorDim(output_tensor, 3);
        LOGD("Model output dims: height=%d, width=%d, channels=%d",
             outputHeight, outputWidth, outputChannels);
    }

    int outputSize = outputHeight * outputWidth * outputChannels;
    float* output_buffer = new float[outputSize];

    status = TfLiteTensorCopyToBuffer(
            output_tensor, output_buffer,
            outputSize * sizeof(float));
    if (status != kTfLiteOk) {
        LOGE("Failed to copy output tensor to output buffer");
        delete[] input_buffer;
        delete[] output_buffer;
        return -1;
    }
    LOGD("Output copied successfully");

    // 檢查輸出
    float sum = 0;
    for (int i = 0; i < outputSize; i++) {
        sum += output_buffer[i];
    }
    LOGD("Output buffer mean: %.6f", sum / outputSize);
    LOGD("Output buffer[0]: %f", output_buffer[0]);

    // 轉換輸出為 mask (只取第一個通道)
    int threshold = 128;
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            int idx = (i * outputWidth + j) * outputChannels;
            float val = output_buffer[idx];  // 取第一個通道
            out[i][j] = (val * 255 > threshold) ? 255 : 0;
        }
    }

    int nonZeroCount = 0;
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            if (out[i][j] > 0) nonZeroCount++;
        }
    }
    LOGD("Mask non-zero pixels: %d / %d", nonZeroCount, outputHeight * outputWidth);

    delete[] input_buffer;
    delete[] output_buffer;

    LOGD("=== doseg END ===");
    return 0;
}

/**
 * 後處理：由分割 mask 與取樣線求出中心線、交點、切線方向、Range Gate 邊界與角度。
 * 1. 把 224×224 mask 放大 / 裁切回 image_w×image_h，並產生中心線 mask。
 * 2. 若外部取樣線 (p_top→p_bottom) 與中心線有交點：以第一個交點為 center。
 *    否則（備援）：取中心線中位點為 center，改以 kFallbackBeamAngleDeg 通過 center 重畫線束。
 * 3. 在 center 以 PCA 取切線 direction；沿線束往上下找血管 mask 邊界（find_RangeGate）。
 * 4. 計算絕對角（切線 vs 垂直軸）與相對角（切線 vs 線束），並沿法向掃描管徑。
 * @param mask     模型輸出 mask（224×224，CV_8UC1）
 * @param img_ori  前處理後的 224×224 灰階影像（只用來決定尺寸）
 * @param p_top    取樣線頂端點（原圖座標）
 * @param p_bottom 取樣線底端點（原圖座標）
 * @param image_h  原圖高
 * @param image_w  原圖寬
 * @return PostProcessResult；中心線為空、取樣線為 (0,0)-(0,0) 或中心線點數 < 20 時 success=false
 */
PostProcessResult SuperResolution::postprocess(const cv::Mat& mask, const cv::Mat& img_ori, cv::Point p_top, cv::Point p_bottom, int image_h, int image_w) {
    PostProcessResult result;
    result.success = false;

    cv::Mat img = img_ori.clone();
    cv::Mat image = mask.clone();

    // ====== 224 -> target size ======
    if (image_h > image_w) {
        img = resize_img(img, image_h);
        image = resize_img(mask, image_h);
        LOGD("Resize mask size: %dx%d (WxH)", image.cols, image.rows);
        int width_pad = (image_h - image_w) / 2;
        image = crop_img(image, width_pad, width_pad + image_w, 0, image_h);
    } else {
        img = resize_img(img, image_w);
        image = resize_img(mask, image_w);
        LOGD("Resize mask size: %dx%d (WxH)", image.cols, image.rows);
        int width_pad = (image_w - image_h) / 2;
        image = crop_img(image, 0, image_w, width_pad, width_pad + image_h);
    }


    // ====== Generate centerline ======
    cv::Mat centerLine = process_single_centerline(img, mask);

    // Crop to final size
    if (image_h > image_w) {
        int width_pad = (image_h - image_w) / 2;
        img = crop_img(img, width_pad, width_pad + image_w, 0, image_h);
        centerLine = crop_img(centerLine, width_pad, width_pad + image_w, 0, image_h);
    } else {
        int width_pad = (image_w - image_h) / 2;
        img = crop_img(img, 0, image_w, width_pad, width_pad + image_h);
        centerLine = crop_img(centerLine, 0, image_w, width_pad, width_pad + image_h);
    }

    LOGD("CenterLine mask size: %dx%d (WxH)", centerLine.cols, centerLine.rows);

    if (cv::countNonZero(centerLine) == 0) {
        LOGD("Warning: skeleton is empty");
        return result;
    }

    if (p_top == cv::Point(0, 0) && p_bottom == cv::Point(0, 0)) {
        LOGD("Failed to detect line segment");
        return result;
    }

    // ====== Draw line beam ======
    cv::Mat lines = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::line(lines, p_top, p_bottom, 255, 1);

    // ====== Find intersection ======
    cv::Mat intersection_mask;
    cv::bitwise_and(lines, centerLine, intersection_mask);
    std::vector<cv::Point> intersection_pts;
    cv::findNonZero(intersection_mask, intersection_pts);

    cv::Point center;
    cv::Point2f direction(1.0f, 0.0f);
    cv::Point in_top, in_bottom;
    cv::Point intersection_top, intersection_bottom;

    LOGD("find %d intersection_pts.", intersection_pts.size());
    if (!intersection_pts.empty()) {
        center = intersection_pts[0];

        cv::Mat mask_and;
        cv::bitwise_and(lines, image, mask_and);
        std::vector<cv::Point> green_points;
        cv::findNonZero(mask_and, green_points);

        if (!green_points.empty()) {
            direction = get_tangent_direction(centerLine, center, 15);

            in_top = *std::min_element(green_points.begin(), green_points.end(),
                                       [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
            in_bottom = *std::max_element(green_points.begin(), green_points.end(),
                                          [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
             intersection_top = find_RangeGate(center, in_top, image);
             intersection_bottom = find_RangeGate(center, in_bottom, image);
        }
    } else {
        // Find center of centerline
        std::vector<cv::Point> cl_points;
        cv::findNonZero(centerLine, cl_points);

        if (static_cast<int>(cl_points.size()) < 20) {
            LOGD("Not enough centerline points");
            return result;
        }

        std::vector<int> xs, ys;
        for (const auto& pt : cl_points) {
            xs.push_back(pt.x);
            ys.push_back(pt.y);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());

        int median_x = xs[xs.size() / 2];
        int median_y = ys[ys.size() / 2];

        int best_idx = 0;
        long long best_dist = std::numeric_limits<long long>::max();
        for (size_t j = 0; j < cl_points.size(); j++) {
            long long d = static_cast<long long>(cl_points[j].x - median_x) *
                    (cl_points[j].x - median_x) +
                    static_cast<long long>(cl_points[j].y - median_y) *
                    (cl_points[j].y - median_y);
            if (d < best_dist) {
                best_dist = d;
                best_idx = static_cast<int>(j);
            }
        }
        center = cl_points[best_idx];

        direction = get_tangent_direction(centerLine, center, 15);

        auto [p_t, p_b] = get_boundary_intersection_direct(
                image.size(), center, kFallbackBeamAngleDeg);
        p_top = p_t;
        p_bottom = p_b;

        lines = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        cv::line(lines, p_top, p_bottom, 255, 1);

        cv::Mat mask_and;
        cv::bitwise_and(lines, image, mask_and);
        std::vector<cv::Point> green_points;
        cv::findNonZero(mask_and, green_points);

        if (!green_points.empty()) {
            in_top = *std::min_element(green_points.begin(), green_points.end(),
                                       [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
            in_bottom = *std::max_element(green_points.begin(), green_points.end(),
                                          [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });

            intersection_top = find_RangeGate(center, in_top, image);
            intersection_bottom = find_RangeGate(center, in_bottom, image);
        }
    }

    // 計算角度
    double angle_abs = calculate_angle_between_vectors(cv::Point(0, 0), cv::Point(0, 1), direction, true);
    double angle_relative = 0.0;
    if (intersection_top != cv::Point(0, 0) && intersection_bottom != cv::Point(0, 0)) {
        angle_relative = calculate_angle_between_vectors(intersection_bottom, intersection_top, direction, true);
    }

    // 計算管徑：從 center 沿血管法向往兩側掃描 mask 邊界
    cv::Point2f perp(-direction.y, direction.x);
    auto walk_to_edge = [&](cv::Point2f dir) -> int {
        int last_inside = 0;
        const int max_search = 100;
        for (int d = 1; d < max_search; d++) {
            int x = static_cast<int>(std::round(center.x + dir.x * d));
            int y = static_cast<int>(std::round(center.y + dir.y * d));
            if (x < 0 || x >= image.cols || y < 0 || y >= image.rows) break;
            if (image.at<uchar>(y, x) == 0) break;
            last_inside = d;
        }
        return last_inside;
    };

    double vessel_width = 0.0;
    cv::Point diameter_top = center;
    cv::Point diameter_bottom = center;
    if (center.x >= 0 && center.x < image.cols &&
        center.y >= 0 && center.y < image.rows &&
        image.at<uchar>(center.y, center.x) != 0) {
        int r1 = walk_to_edge(perp);
        int r2 = walk_to_edge(cv::Point2f(-perp.x, -perp.y));
        vessel_width = static_cast<double>(r1 + r2);
        diameter_top = cv::Point(
                static_cast<int>(std::round(center.x - perp.x * r2)),
                static_cast<int>(std::round(center.y - perp.y * r2)));
        diameter_bottom = cv::Point(
                static_cast<int>(std::round(center.x + perp.x * r1)),
                static_cast<int>(std::round(center.y + perp.y * r1)));
    }


// 填充結果
    result.success = true;
    result.angle_abs = angle_abs;
    result.angle_relative = angle_relative;
    result.center = center;
    result.intersection_top = intersection_top;
    result.intersection_bottom = intersection_bottom;
    result.p_top = p_top;
    result.p_bottom = p_bottom;
    result.direction = direction;
    result.vessel_diameter = vessel_width;
    result.diameter_top = diameter_top;
    result.diameter_bottom = diameter_bottom;
    result.perp_direction = perp;
    return result;
}

/**
 * 把後處理結果疊畫到原圖：綠色線束（畫到管徑 kRangeGateRatio 處）、紅色切線、
 * 紅色垂直短線標示 Range Gate 上下位置。
 * @param img_ori 原圖（灰階或 BGR）
 * @param result  postprocess() 的結果；success=false 時只回傳轉成 BGR 的原圖
 * @return BGR 視覺化影像
 */
cv::Mat SuperResolution::visualizePostProcess(
    const cv::Mat& img_ori,
    const PostProcessResult& result) {

    // 準備結果圖像
    cv::Mat result_img;
    if (img_ori.channels() == 1) {
        cv::cvtColor(img_ori, result_img, cv::COLOR_GRAY2BGR);
    } else {
        result_img = img_ori.clone();
    }

    if (!result.success) {
        return result_img;
    }

    // 計算管徑 2/3 位置 (往中心線靠近)
    cv::Point range_top = result.center + (result.intersection_top - result.center) * kRangeGateRatio;
    cv::Point range_bottom = result.center + (result.intersection_bottom - result.center) * kRangeGateRatio;

    // 繪製線束（綠色）：畫到管徑 2/3 處
    cv::line(result_img, result.p_top, range_top, cv::Scalar(0, 255, 0), 1);
    cv::line(result_img, range_bottom, result.p_bottom, cv::Scalar(0, 255, 0), 1);

    // 繪製切線（紅色）
    draw_tangent(result_img, result.center, result.direction, 30);

    // 繪製 Range Gate（紅色垂直線）：標在管徑 2/3 處
    if (result.intersection_top != cv::Point(0, 0) && result.intersection_bottom != cv::Point(0, 0)) {
        draw_perpendicular_line(result_img, result.p_top, result.p_bottom,
                                        range_top, 20, cv::Scalar(0, 0, 255), 2);
        draw_perpendicular_line(result_img, result.p_top, result.p_bottom,
                                        range_bottom, 20, cv::Scalar(0, 0, 255), 2);
    }

    return result_img;
}

/**
 * 整條 pipeline 的進入點（由 JNI superResolutionFromJNI 呼叫）：
 * 1. ARGB → BGR Mat　2. 由 top_x / line_angle 算出取樣線端點　3. resize_with_padding 到 224
 * 4. 轉灰階　5. CLAHE　6. doseg 模型推論　7. postprocess　8. mask 放大回原尺寸
 * 9. visualizePostProcess 疊圖並轉回 ARGB。後處理結果同時存入 last_result_ 供 PW_* getter 讀取。
 * @param lr_img_rgb inputWidth×inputHeight 的 ARGB 像素陣列（row-major）
 * @param line_angle 取樣線角度（度，見 get_line_point 的定義）
 * @param top_x      取樣線頂端 x 座標
 * @return 同尺寸的 ARGB 視覺化影像；tensor 配置或推論失敗時回傳 nullptr
 */
std::unique_ptr<int[]> SuperResolution::DoSuperResolution(int* lr_img_rgb, double line_angle, int top_x) {

    // ============= 前處理 =============
    LOGD("=== DoSuperResolution START ===");
    LOGD("Target size: %dx%d (WxH)", inputWidth, inputHeight);
    std::vector<int> result_values;

    // 1. ARGB → BGR
    LOGD("Step 1: Converting to Mat...");
    cv::Mat bgr_img(inputHeight, inputWidth, CV_8UC3);
    for (int i = 0; i < inputHeight; i++) {
        for (int j = 0; j < inputWidth; j++) {
            int argb = lr_img_rgb[i * inputWidth + j];
            uint8_t r = (argb >> 16) & 0xff;
            uint8_t g = (argb >> 8) & 0xff;
            uint8_t b = argb & 0xff;
            bgr_img.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
        }
    }

    {
        cv::Scalar mean, stddev;
        cv::meanStdDev(bgr_img, mean, stddev);
        LOGD("Step1 bgr - mean(B,G,R): (%.2f,%.2f,%.2f)", mean[0], mean[1], mean[2]);
    }

    // 2. 取得線段端點
    LOGD("Step 2: Get line points...");
    cv::Point p_top(top_x, 0);
    cv::Point p_bottom = get_line_point(p_top.x, p_top.y, line_angle, inputHeight);

    // 3. Resize with padding to 224x224
    LOGD("Step 3: Resize with padding to 224...");
    cv::Mat padded = resize_with_padding(bgr_img , 224);

    {
        cv::Scalar mean, stddev;
        cv::meanStdDev(padded, mean, stddev);
        LOGD("Step3 padded - mean(B,G,R): (%.2f,%.2f,%.2f)", mean[0], mean[1], mean[2]);
        LOGD("padded size: %dx%d (WxH)", padded.cols, padded.rows);
    }

    // 4. 轉灰階
    LOGD("Step 4: Convert to gray...");
    cv::Mat gray;
    cv::cvtColor(padded, gray, cv::COLOR_BGR2GRAY);

    {
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        LOGD("Step4 gray - mean: %.2f, std: %.2f", mean[0], stddev[0]);
    }

    // 5. CLAHE
    LOGD("Step 5: Apply CLAHE...");
    cv::Mat enhanced = apply_clahe(gray);

    {
        cv::Scalar mean, stddev;
        cv::meanStdDev(enhanced, mean, stddev);
        LOGD("Step5 enhanced - mean: %.2f, std: %.2f", mean[0], stddev[0]);
    }

    // ============= 模型預測 =============
    LOGD("Step 6: Running model inference...");

    TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
    if (status != kTfLiteOk) {
        LOGE("Failed to allocate tensors!");
        return nullptr;
    }

    // 分配輸出 mask 的記憶體 (224x224)
    int modelOutHeight = 224;
    int modelOutWidth = 224;
    int** model_out = new int*[modelOutHeight];
    for (int i = 0; i < modelOutHeight; i++) {
        model_out[i] = new int[modelOutWidth];
    }

    // 呼叫模型推論
    int seg_status = doseg(enhanced, model_out, interpreter_, false);
    if (seg_status == -1) {
        LOGE("Model inference failed!");
        return nullptr;
    }

    // 將 mask 轉成 cv::Mat
    cv::Mat mask(modelOutHeight, modelOutWidth, CV_8UC1);
    for (int i = 0; i < modelOutHeight; i++) {
        for (int j = 0; j < modelOutWidth; j++) {
            mask.at<uchar>(i, j) = (uchar)model_out[i][j];
        }
    }

    // 輸出 mask 資訊
    int whiteCount = 0;
    int minX = 224, maxX = 0, minY = 224, maxY = 0;
    for (int i = 0; i < modelOutHeight; i++) {
        for (int j = 0; j < modelOutWidth; j++) {
            if (mask.at<uchar>(i, j) > 0) {
                whiteCount++;
                if (i < minY) minY = i;
                if (i > maxY) maxY = i;
                if (j < minX) minX = j;
                if (j > maxX) maxX = j;
            }
        }
    }
    LOGD("=== Mask Output ===");
    LOGD("White pixels: %d / %d", whiteCount, modelOutHeight * modelOutWidth);
    LOGD("Target size: %dx%d (WxH)", mask.cols, mask.rows);
    if (whiteCount > 0) {
        LOGD("Y range: %d~%d, X range: %d~%d", minY, maxY, minX, maxX);
    } else {
        LOGD("Mask is all black!");
    }

    cv::Mat mask_output = mask;

    // ============= 後處理 =============
    LOGD("Step 7: postprocess");
    PostProcessResult result = postprocess(mask_output, enhanced, cv::Point(top_x, 0), p_bottom, inputHeight, inputWidth);


    if (result.success) {
        LOGD("=== PostProcess Results ===");
        LOGD("Absolute angle: %.2f", result.angle_abs);
        LOGD("Relative angle: %.2f", result.angle_relative);
        LOGD("Center point: (%d, %d)", result.center.x, result.center.y);
        LOGD("Intersection top: (%d, %d)", result.intersection_top.x, result.intersection_top.y);
        LOGD("Intersection bottom: (%d, %d)", result.intersection_bottom.x, result.intersection_bottom.y);
        LOGD("Vessel width at green line: %.2f pixels", result.vessel_diameter);
    }

    // 8. mask放大回原始尺寸
    LOGD("Step 8: Resize back to %dx%d...", inputWidth, inputHeight);
    cv::Mat final_mask;
    if (inputHeight > inputWidth) {
        final_mask = resize_img(mask, inputHeight);
        int width_pad = (inputHeight - inputWidth) / 2;
        final_mask = crop_img(final_mask, width_pad, width_pad + inputWidth, 0, inputHeight);
    } else {
        final_mask = resize_img(mask, inputWidth);
        int width_pad = (inputWidth - inputHeight) / 2;
        final_mask = crop_img(final_mask, 0, inputWidth, width_pad, width_pad + inputHeight);
    }

    if (final_mask.empty()) {
        LOGE("final_output is empty after resize!");
        final_mask = cv::Mat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    ////////////////////////////////// 測試結果用(可省略)
    // 視覺化
    cv::Mat visualize_img = visualizePostProcess(bgr_img, result);
    if (visualize_img.empty()) {
        LOGE("final_img is empty! Using original image instead.");
        visualize_img = bgr_img.clone();
    }
    LOGD("final_img size: %dx%d, channels: %d", visualize_img.cols, visualize_img.rows, visualize_img.channels());
    //////////////////////////////////

    if (visualize_img.channels() == 1) {
        LOGE("final_img is 1 channels");
        cv::cvtColor(visualize_img, visualize_img, cv::COLOR_GRAY2BGR);
    }

    // 9. 轉成 int array
    LOGD("Step 9: Converting to int array...");

    int outputPixelNumber = visualize_img.rows * visualize_img.cols;
    auto result_array = std::make_unique<int[]>(outputPixelNumber);

    for (int i = 0; i < outputPixelNumber; i++) {
        int h = i / visualize_img.cols;
        int w = i % visualize_img.cols;

        cv::Vec3b pixel = visualize_img.at<cv::Vec3b>(h, w);

        uchar b = pixel[0];
        uchar g = pixel[1];
        uchar r = pixel[2];

        result_array[i] =
                (255u << 24) |   // Alpha
                (r << 16)    |   // Red
                (g << 8)     |   // Green
                (b);             // Blue
    }

    last_result_ = result;

    return result_array;
}

}  // namespace superresolution
}  // namespace examples
}  // namespace tflite