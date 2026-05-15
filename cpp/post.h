#ifndef POST_H
#define POST_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

struct FrameData {
    std::vector<cv::Point> line;
    cv::Mat raw_mask;
    cv::Mat img;
};

void post_process(std::map<std::string, FrameData>& frames,
                  int image_h,
                  int image_w,
                  const std::string& output_dir);

#endif