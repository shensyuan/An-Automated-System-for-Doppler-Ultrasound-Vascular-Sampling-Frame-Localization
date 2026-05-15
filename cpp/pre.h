#ifndef PRE_H
#define PRE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct PreprocessResult {
    std::vector<cv::Mat> no_ui_frames;
    std::vector<std::vector<cv::Point>> line_frames;
    std::vector<std::string> file_names;

    int image_h;
    int image_w;
};

cv::Mat apply_clahe(const cv::Mat& gray);

cv::Mat remove_green_red(const cv::Mat& img);

cv::Mat resize_with_padding(const cv::Mat& img, int size);

std::pair<double, int> detected_line(const cv::Mat& img);

cv::Point get_line_point(
    int top_x,
    int top_y,
    double angle_deg,
    int length
);

PreprocessResult extract_frames_with_timestamp(
    const std::string& input_dir
);

#endif