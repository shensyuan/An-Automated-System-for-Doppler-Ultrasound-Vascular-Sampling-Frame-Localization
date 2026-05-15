#include "pre.h"

#include <filesystem>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;

// ==================== CLAHE ====================

cv::Mat apply_clahe(const cv::Mat& gray) {

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        2.0,
        cv::Size(8, 8)
    );

    cv::Mat enhanced;
    clahe->apply(gray, enhanced);

    return enhanced;
}

// ==================== Remove Green / Red ====================

cv::Mat remove_green_red(const cv::Mat& img) {

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // ===== Green =====
    cv::Scalar lower_green(40, 80, 80);
    cv::Scalar upper_green(80, 255, 255);

    cv::Mat mask_green;
    cv::inRange(hsv, lower_green, upper_green, mask_green);

    // ===== Red =====
    cv::Scalar lower_red1(0, 80, 80);
    cv::Scalar upper_red1(10, 255, 255);

    cv::Scalar lower_red2(170, 80, 80);
    cv::Scalar upper_red2(180, 255, 255);

    cv::Mat mask_red1, mask_red2;

    cv::inRange(hsv, lower_red1, upper_red1, mask_red1);
    cv::inRange(hsv, lower_red2, upper_red2, mask_red2);

    cv::Mat mask_red;
    cv::bitwise_or(mask_red1, mask_red2, mask_red);

    cv::Mat mask;
    cv::bitwise_or(mask_green, mask_red, mask);

    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 2);

    cv::Mat result;

    cv::inpaint(img, mask, result, 5, cv::INPAINT_TELEA);

    return result;
}

// ==================== Resize With Padding ====================

cv::Mat resize_with_padding(const cv::Mat& img, int size) {

    int h = img.rows;
    int w = img.cols;

    double scale = static_cast<double>(size) /
                   static_cast<double>(std::max(h, w));

    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat resized;

    cv::resize(
        img,
        resized,
        cv::Size(new_w, new_h)
    );

    int pad_top = (size - new_h) / 2;
    int pad_bottom = size - new_h - pad_top;

    int pad_left = (size - new_w) / 2;
    int pad_right = size - new_w - pad_left;

    cv::Mat padded;

    cv::copyMakeBorder(
        resized,
        padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );

    return padded;
}

// ==================== Detect Doppler Beam ====================

std::pair<double, int> detected_line(const cv::Mat& img) {

    cv::Scalar lower_green(30, 130, 30);
    cv::Scalar upper_green(95, 255, 255);

    cv::Mat mask;

    cv::inRange(img, lower_green, upper_green, mask);

    std::vector<cv::Point> points;

    cv::findNonZero(mask, points);

    if (points.empty()) {
        return {9999.0, -1};
    }

    auto p_top = *std::min_element(
        points.begin(),
        points.end(),
        [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
        }
    );

    auto p_bottom = *std::max_element(
        points.begin(),
        points.end(),
        [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
        }
    );

    double dx = static_cast<double>(p_bottom.x - p_top.x);
    double dy = static_cast<double>(p_bottom.y - p_top.y);

    double alpha_rad = std::atan(std::abs(dx / dy));

    double alpha_deg = alpha_rad * 180.0 / CV_PI;

    if (dx * dy < 0) {
        return {alpha_deg, p_top.x};
    }

    return {-alpha_deg, p_top.x};
}

// ==================== Get Line Point ====================

cv::Point get_line_point(
    int top_x,
    int top_y,
    double angle_deg,
    int length
) {

    double angle_rad = angle_deg * CV_PI / 180.0;

    double dx = std::sin(angle_rad) * length;
    double dy = std::cos(angle_rad) * length;

    int bottom_x = static_cast<int>(std::round(top_x + dx));
    int bottom_y = static_cast<int>(std::round(top_y + dy));

    return cv::Point(bottom_x, bottom_y);
}

// ==================== Main Preprocess ====================

PreprocessResult extract_frames_with_timestamp(
    const std::string& input_dir
) {

    PreprocessResult result;

    std::vector<std::string> image_files;

    for (const auto& entry : fs::directory_iterator(input_dir)) {

        if (entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }

    std::cout
        << "Found "
        << image_files.size()
        << " images"
        << std::endl;

    for (const auto& path : image_files) {

        cv::Mat frame = cv::imread(path);

        if (frame.empty()) {
            continue;
        }

        result.image_h = frame.rows;
        result.image_w = frame.cols;

        std::string filename =
            fs::path(path).filename().string();

        std::vector<cv::Point> line(2);

        auto [line_angle, top_x] =
            detected_line(frame);

        std::cout
            << filename
            << " angle: "
            << line_angle
            << " top_x: "
            << top_x
            << std::endl;

        if (top_x < 0) {

            line[0] = cv::Point(0, 0);
            line[1] = cv::Point(0, 0);

        } else {

            line[0] = cv::Point(top_x, 0);

            line[1] = get_line_point(
                line[0].x,
                line[0].y,
                line_angle,
                result.image_h
            );
        }

        cv::Mat removed =
            remove_green_red(frame);

        cv::Mat resized =
            resize_with_padding(removed, 224);

        cv::Mat gray;

        cv::cvtColor(
            resized,
            gray,
            cv::COLOR_BGR2GRAY
        );

        cv::Mat enhanced =
            apply_clahe(gray);

        result.no_ui_frames.push_back(enhanced);

        result.line_frames.push_back(line);

        result.file_names.push_back(filename);
    }

    return result;
}