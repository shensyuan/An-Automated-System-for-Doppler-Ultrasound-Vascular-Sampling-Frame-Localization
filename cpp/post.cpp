#include "post.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <chrono>
#include <numeric>

// ==================== Utility Functions ====================

cv::Mat resize_img(const cv::Mat& img, int size) {
    int h = img.rows;
    int w = img.cols;
    double scale = static_cast<double>(size) / std::min(h, w);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    return resized;
}

cv::Mat crop_img(const cv::Mat& img, int x1, int x2, int y1, int y2) {
    return img(cv::Range(y1, y2), cv::Range(x1, x2)).clone();
}

// Zhang-Suen thinning algorithm for skeletonization
void thinningZhangSuen(const cv::Mat& src, cv::Mat& dst) {
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

bool is_valid_centerline(const std::vector<int>& x_pts, const std::vector<int>& y_pts) {
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

// Catmull-Rom spline point evaluation
cv::Point2f catmullRomPoint(const cv::Point2f& p0, const cv::Point2f& p1,
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

std::vector<cv::Point2f> generateSplinePoints(const std::vector<cv::Point2f>& controlPoints, int totalPoints) {
    std::vector<cv::Point2f> result;
    int n = static_cast<int>(controlPoints.size());
    if (n < 2) return result;

    if (n == 2) {
        // Linear interpolation (k=1)
        for (int i = 0; i < totalPoints; i++) {
            float t = static_cast<float>(i) / (totalPoints - 1);
            float x = controlPoints[0].x + t * (controlPoints[1].x - controlPoints[0].x);
            float y = controlPoints[0].y + t * (controlPoints[1].y - controlPoints[0].y);
            result.push_back(cv::Point2f(x, y));
        }
        return result;
    }

    // For 3 points: treat as 2 Catmull-Rom segments (equivalent to quadratic)
    // For 4+ points: standard Catmull-Rom (cubic)
    int segments = n - 1;
    int pointsPerSegment = totalPoints / segments;
    int extra = totalPoints - pointsPerSegment * segments;

    int pointIdx = 0;
    for (int seg = 0; seg < segments; seg++) {
        cv::Point2f p0, p1, p2, p3;
        p1 = controlPoints[seg];
        p2 = controlPoints[seg + 1];

        if (seg == 0) {
            p0 = p1 - (p2 - p1);  // extrapolate
        } else {
            p0 = controlPoints[seg - 1];
        }

        if (seg + 2 < n) {
            p3 = controlPoints[seg + 2];
        } else {
            p3 = p2 + (p2 - p1);  // extrapolate
        }

        int nPts = pointsPerSegment + (seg < extra ? 1 : 0);
        for (int j = 0; j < nPts; j++) {
            float t = static_cast<float>(j) / nPts;
            result.push_back(catmullRomPoint(p0, p1, p2, p3, t));
            pointIdx++;
        }
    }

    return result;
}

cv::Mat process_single_centerline(const cv::Mat& img_orig, const cv::Mat& mask_224) {
    int h_orig = img_orig.rows;
    int w_orig = img_orig.cols;
    double scale_x = static_cast<double>(w_orig) / 224.0;
    double scale_y = static_cast<double>(h_orig) / 224.0;

    cv::Mat centerline_mask = cv::Mat::zeros(h_orig, w_orig, CV_8UC1);

    // Binary threshold
    cv::Mat binary_mask;
    cv::threshold(mask_224, binary_mask, 127, 255, cv::THRESH_BINARY);

    // Connected components labeling
    cv::Mat labeled_array;
    int num_features = cv::connectedComponents(binary_mask, labeled_array, 8, CV_32S);

    for (int i = 1; i <= num_features; i++) {
        // Single region mask
        cv::Mat single_region = (labeled_array == i);
        if (cv::countNonZero(single_region) < 20) continue;

        // Euclidean distance transform
        cv::Mat dist_map;
        cv::distanceTransform(single_region, dist_map, cv::DIST_L2, cv::DIST_MASK_PRECISE);

        // Skeletonization via thinning
        cv::Mat skel_region;
        thinningZhangSuen(single_region, skel_region);

        // Get skeleton points
        std::vector<cv::Point> skel_points;
        cv::findNonZero(skel_region, skel_points);

        // For each x, keep the (y, dist_val) with maximum distance value
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

        // Sort by x
        std::vector<int> sorted_x, sorted_y;
        for (const auto& kv : region_best_pts) {
            sorted_x.push_back(kv.first);
            sorted_y.push_back(kv.second.first);
        }

        if (!is_valid_centerline(sorted_x, sorted_y)) continue;

        // Scale coordinates
        std::vector<cv::Point2f> pts_scaled;
        for (size_t j = 0; j < sorted_x.size(); j++) {
            pts_scaled.push_back(cv::Point2f(
                sorted_x[j] * scale_x,
                sorted_y[j] * scale_y
            ));
        }

        // Spline fitting and sampling
        std::vector<cv::Point2f> fine_pts = generateSplinePoints(pts_scaled, 2000);

        // Draw onto centerline mask
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

// ==================== Line / Range Gate Functions ====================

std::pair<cv::Point, cv::Point> draw_perpendicular_line(cv::Mat& image, cv::Point line_p1, cv::Point line_p2,
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

cv::Point find_RangeGate(cv::Point start_pt, cv::Point target_pt, const cv::Mat& img) {
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

std::pair<cv::Point, cv::Point> get_boundary_intersection_direct(cv::Size mask_shape, cv::Point center_pt, double angle_deg) {
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

double get_absolute_angle(const cv::Point2f& v, double relative_angle_deg) {
    double base_angle_rad = std::atan2(static_cast<double>(v.y), static_cast<double>(v.x));
    double relative_angle_rad = relative_angle_deg * CV_PI / 180.0;
    double final_angle_rad = base_angle_rad + relative_angle_rad;
    double final_angle_deg = final_angle_rad * 180.0 / CV_PI;
    final_angle_deg = std::fmod(final_angle_deg, 360.0);
    if (final_angle_deg < 0) final_angle_deg += 360.0;
    return final_angle_deg;
}

cv::Point2f get_tangent_direction(const cv::Mat& skeleton, cv::Point point, int window_size) {
    int x0 = point.x;
    int y0 = point.y;
    int h = skeleton.rows;
    int w = skeleton.cols;

    if (skeleton.empty()) {
        std::cout << "Warning: skeleton is empty" << std::endl;
        return cv::Point2f(1.0f, 0.0f);
    }

    if (!(x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)) {
        std::cout << "Warning: point (" << x0 << ", " << y0 << ") out of image range" << std::endl;
        return cv::Point2f(1.0f, 0.0f);
    }

    if (skeleton.at<uchar>(y0, x0) == 0) {
        std::cout << "Warning: point (" << x0 << ", " << y0 << ") not on skeleton" << std::endl;
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
        std::cout << "Warning: only " << roi_points.size() << " skeleton points near " << point << std::endl;
        return cv::Point2f(1.0f, 0.0f);
    }

    // Convert to full-image coordinates and build matrix for PCA
    cv::Mat points_mat(static_cast<int>(roi_points.size()), 2, CV_32F);
    for (size_t i = 0; i < roi_points.size(); i++) {
        points_mat.at<float>(static_cast<int>(i), 0) = static_cast<float>(roi_points[i].x + x1);
        points_mat.at<float>(static_cast<int>(i), 1) = static_cast<float>(roi_points[i].y + y1);
    }

    // Compute mean and center
    cv::Mat mean;
    cv::reduce(points_mat, mean, 0, cv::REDUCE_AVG);
    cv::Mat centered = points_mat - cv::repeat(mean, points_mat.rows, 1);

    // Check if all points are the same
    if (cv::norm(centered, cv::NORM_L2) < 1e-6f) {
        std::cout << "Warning: all points coincide near " << point << std::endl;
        return cv::Point2f(1.0f, 0.0f);
    }

    // SVD for PCA
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

double calculate_angle_between_vectors(cv::Point p1, cv::Point p2, const cv::Point2f& v_given, bool absolute) {
    cv::Point2f vec_a(
        static_cast<float>(p2.x - p1.x),
        static_cast<float>(p2.y - p1.y)
    );
    cv::Point2f vec_b = v_given;

    float norm_a = std::sqrt(vec_a.x * vec_a.x + vec_a.y * vec_a.y);
    float norm_b = std::sqrt(vec_b.x * vec_b.x + vec_b.y * vec_b.y);

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0;

    float dot = vec_a.x * vec_b.x + vec_a.y * vec_b.y;
    float cos_theta = dot / (norm_a * norm_b);
    cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));

    double angle_rad = std::acos(static_cast<double>(cos_theta));
    double angle_deg = angle_rad * 180.0 / CV_PI;

    if (absolute) {
        if (angle_deg > 90) {
            angle_deg = angle_deg - 180;
        }
    }

    return angle_deg;
}

void draw_tangent(cv::Mat& img, cv::Point point, const cv::Point2f& direction, int length) {
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

// ==================== Main Processing Function ====================

void post_process(std::map<std::string, FrameData>& frames, int image_h, int image_w, const std::string& output_dir) {
    // Create output directory
    std::string cmd = "mkdir -p \"" + output_dir + "\"";
    system(cmd.c_str());

    std::cout << "Total " << frames.size() << " images to process..." << std::endl;

    for (auto& [fname, frame] : frames) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat line_mask = frame.line.empty() ? cv::Mat() : cv::Mat();  // not used directly
        cv::Mat mask = frame.raw_mask.clone();
        cv::Mat img_ori = frame.img.clone();

        if (frame.line.empty() || mask.empty()) {
            std::cout << "Skipping " << fname << std::endl;
            continue;
        }

        cv::Point p_top = frame.line[0];
        cv::Point p_bottom = frame.line[1];

        if (img_ori.empty()) {
            std::cout << "Cannot find original image for " << fname << std::endl;
            img_ori = mask.clone();
        }

        // ====== 224 -> target size ======
        cv::Mat image;  // resized/cropped version of mask
        if (image_h > image_w) {
            img_ori = resize_img(img_ori, image_h);
            image = resize_img(mask, image_h);
            int width_pad = (image_h - image_w) / 2;
            image = crop_img(image, width_pad, width_pad + image_w, 0, image_h);
        } else {
            img_ori = resize_img(img_ori, image_w);
            image = resize_img(mask, image_w);
            int width_pad = (image_w - image_h) / 2;
            image = crop_img(image, 0, image_w, width_pad, width_pad + image_h);
        }

        // ====== Generate centerline ======
        cv::Mat centerLine = process_single_centerline(img_ori, mask);

        // Crop img_ori and centerLine to final size
        if (image_h > image_w) {
            int width_pad = (image_h - image_w) / 2;
            img_ori = crop_img(img_ori, width_pad, width_pad + image_w, 0, image_h);
            centerLine = crop_img(centerLine, width_pad, width_pad + image_w, 0, image_h);
        } else {
            int width_pad = (image_w - image_h) / 2;
            img_ori = crop_img(img_ori, 0, image_w, width_pad, width_pad + image_h);
            centerLine = crop_img(centerLine, 0, image_w, width_pad, width_pad + image_h);
        }

        if (cv::countNonZero(centerLine) == 0) {
            std::cout << "Warning: " << fname << " skeleton is empty, skipping." << std::endl;
            continue;
        }

        if (p_top == cv::Point(0, 0) && p_bottom == cv::Point(0, 0)) {
            std::cout << fname << " failed to detect line segment" << std::endl;
            continue;
        }

        // ====== Draw line beam ======
        cv::Mat lines = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        cv::line(lines, p_top, p_bottom, 255, 1);

        // ====== Find intersection between line and centerline ======
        cv::Mat intersection_mask;
        cv::bitwise_and(lines, centerLine, intersection_mask);
        std::vector<cv::Point> intersection_pts;
        cv::findNonZero(intersection_mask, intersection_pts);

        cv::Point center;
        cv::Point2f direction(1.0f, 0.0f);
        cv::Point in_top, in_bottom;
        cv::Point intersection_top, intersection_bottom;

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
            // ====== Find center of centerline ======
            std::vector<cv::Point> cl_points;
            cv::findNonZero(centerLine, cl_points);

            if (static_cast<int>(cl_points.size()) < 20) {
                // In Python, this returns None, None — skip saving
                std::cout << fname << " not enough centerline points" << std::endl;
                continue;
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

            // Find nearest skeleton point to the median
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

            // Default line beam at 60 degrees relative to tangent
            auto [p_t, p_b] = get_boundary_intersection_direct(
                image.size(), center, get_absolute_angle(direction, 60));
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

        // ====== Visualization ======
        cv::Mat result;
        if (img_ori.channels() == 1) {
            cv::cvtColor(img_ori, result, cv::COLOR_GRAY2BGR);
        } else {
            result = img_ori.clone();
        }

        // Draw skeleton (green overlay)
        cv::Mat skeleton_color;
        cv::cvtColor(centerLine, skeleton_color, cv::COLOR_GRAY2BGR);
        skeleton_color.setTo(cv::Scalar(0, 255, 0), centerLine);
        cv::addWeighted(result, 0.7, skeleton_color, 0.3, 0, result);

        // Draw beam (green lines)
        cv::line(result, p_top, intersection_top, cv::Scalar(0, 255, 0), 1);
        cv::line(result, intersection_bottom, p_bottom, cv::Scalar(0, 255, 0), 1);

        // Draw angel
        draw_tangent(result, center, direction, 30);

        // Draw Range Gate (red perpendicular line)
        if (intersection_top != cv::Point(0, 0) && intersection_bottom != cv::Point(0, 0)) {
            draw_perpendicular_line(result, p_top, p_bottom, intersection_top, 20, cv::Scalar(0, 0, 255), 2);
            draw_perpendicular_line(result, p_top, p_bottom, intersection_bottom, 20, cv::Scalar(0, 0, 255), 2);
        }


        double angle_val = calculate_angle_between_vectors(intersection_bottom, intersection_top, direction, true);
        double angle_abs_val = calculate_angle_between_vectors(cv::Point(0, 0), cv::Point(0, 1), direction, true);

        // Save result
        std::string save_path = output_dir + "/" + fname;
        cv::imwrite(save_path, result);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << fname << " | Angle: " << angle_abs_val << " | Angle relative: " << angle_val
                  << " | center: (" << center.x << ", " << center.y << ")"
                  << " | time: " << elapsed << "s" << std::endl;
    }

    std::cout << "\n=== All processing complete ===" << std::endl;
}

// ==================== Example main() — adapt to your pipeline ====================
/*
int main() {
    // Example usage:
    // Build the frames map from your pre-processing pipeline
    std::map<std::string, FrameData> frames;
    // ... populate frames ...
    // post_process(frames, 900, 700, "./result");
    return 0;
}
*/
