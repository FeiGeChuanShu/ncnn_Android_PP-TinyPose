// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"
static int argmax(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, std::vector<float> &prob)
{
    int size = bottom_blob.total();
    const float* ptr = bottom_blob;
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(ptr[i], i);
    }
    top_blob.create(bottom_blob.c, 1, 1, 4u);
    float* outptr = top_blob;

    for (size_t i = 0; i < bottom_blob.c; i++)
    {
        int size0 = bottom_blob.channel(i).total();
        std::partial_sort(vec.begin()+size0*i, vec.begin() + size0*(i+1), vec.begin() + size0 * (i + 1),
                          std::greater<std::pair<float, int> >());
        outptr[i] = vec[size0 * i].second- size0 * i;
        prob.push_back(vec[size0 * i].first);
    }

    return 0;
}
static void dark_parse(const ncnn::Mat& heatmap,std::vector<int>& dim,std::vector<float>& coords,int px,int py,int ch)
{
    /*DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).
    1) offset = - hassian.inv() * derivative
    2) dx = (heatmap[x+1] - heatmap[x-1])/2.
    3) dxx = (dx[x+1] - dx[x-1])/2.
    4) derivative = Mat([dx, dy])
    5) hassian = Mat([[dxx, dxy], [dxy, dyy]])
    */

    float* heatmap_data = (float*)heatmap.channel(ch).data;
    std::vector<float> heatmap_ch;
    heatmap_ch.insert(heatmap_ch.begin(), heatmap_data, heatmap_data + heatmap.channel(ch).total());

    cv::Mat heatmap_mat = cv::Mat(heatmap_ch).reshape(0, dim[2]);
    heatmap_mat.convertTo(heatmap_mat, CV_32FC1);
    cv::GaussianBlur(heatmap_mat, heatmap_mat, cv::Size(3, 3), 0, 0);
    heatmap_mat = heatmap_mat.reshape(1, 1);
    heatmap_ch = std::vector<float>(heatmap_mat.reshape(1, 1));

    float epsilon = 1e-10;
    //sample heatmap to get values in around target location
    float xy = log(fmax(heatmap_ch[py * dim[3] + px], epsilon));
    float xr = log(fmax(heatmap_ch[py * dim[3] + px + 1], epsilon));
    float xl = log(fmax(heatmap_ch[py * dim[3] + px - 1], epsilon));

    float xr2 = log(fmax(heatmap_ch[py * dim[3] + px + 2], epsilon));
    float xl2 = log(fmax(heatmap_ch[py * dim[3] + px - 2], epsilon));
    float yu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px], epsilon));
    float yd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px], epsilon));
    float yu2 = log(fmax(heatmap_ch[(py + 2) * dim[3] + px], epsilon));
    float yd2 = log(fmax(heatmap_ch[(py - 2) * dim[3] + px], epsilon));
    float xryu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px + 1], epsilon));
    float xryd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px + 1], epsilon));
    float xlyu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px - 1], epsilon));
    float xlyd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px - 1], epsilon));

    //compute dx/dy and dxx/dyy with sampled values
    float dx = 0.5 * (xr - xl);
    float dy = 0.5 * (yu - yd);
    float dxx = 0.25 * (xr2 - 2 * xy + xl2);
    float dxy = 0.25 * (xryu - xryd - xlyu + xlyd);
    float dyy = 0.25 * (yu2 - 2 * xy + yd2);

    //finally get offset by derivative and hassian, which combined by dx/dy and dxx/dyy
    if (dxx * dyy - dxy * dxy != 0)
    {
        float M[2][2] = { dxx, dxy, dxy, dyy };
        float D[2] = { dx, dy };
        cv::Mat hassian(2, 2, CV_32F, M);
        cv::Mat derivative(2, 1, CV_32F, D);
        cv::Mat offset = -hassian.inv() * derivative;
        coords[ch * 2] += offset.at<float>(0, 0);
        coords[ch * 2 + 1] += offset.at<float>(1, 0);
    }
}
static std::vector<float> get_final_preds(const ncnn::Mat& heatmap, const ncnn::Mat& argmax_out)
{
    std::vector<float> coords((size_t)heatmap.c*2);
    for (int i = 0; i < heatmap.c; i++)
    {
        int idx = argmax_out[i];
        coords[i * 2] = idx % heatmap.w;
        coords[i * 2 + 1] = idx / heatmap.w;

        int px = int(coords[i * 2] + 0.5);
        int py = int(coords[i * 2 + 1] + 0.5);

        std::vector<int> dim({ 1, heatmap.c, heatmap.h, heatmap.w });
        dark_parse(heatmap, dim, coords, px, py, i);
    }

    return coords;
}
NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    posenet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    posenet.opt = ncnn::Option();

#if NCNN_VULKAN
    posenet.opt.use_vulkan_compute = use_gpu;
#endif

    posenet.opt.num_threads = ncnn::get_big_cpu_count();
    posenet.opt.blob_allocator = &blob_pool_allocator;
    posenet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    posenet.load_param(parampath);
    posenet.load_model(modelpath);

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    posenet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    posenet.opt = ncnn::Option();

#if NCNN_VULKAN
    posenet.opt.use_vulkan_compute = use_gpu;
#endif

    posenet.opt.num_threads = ncnn::get_big_cpu_count();
    posenet.opt.blob_allocator = &blob_pool_allocator;
    posenet.opt.workspace_allocator = &workspace_pool_allocator;
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    posenet.load_param(mgr,parampath);
    posenet.load_model(mgr,modelpath);
    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::detect(const cv::Mat& rgb, std::vector<KeyPoint>& keypoints, float prob_threshold, float nms_threshold)
{
    //TODO:add human detect model
    return 0;
}
void NanoDet::detect_pose(cv::Mat& rgb, std::vector<KeyPoint>& keypoints)
{
    int w = rgb.cols;
    int h = rgb.rows;
    cv::Mat faceRoiImage = rgb.clone();
    ncnn::Extractor ex_face = posenet.create_extractor();
    ncnn::Mat ncnn_in;
    if(target_size == 128)
        ncnn_in = ncnn::Mat::from_pixels_resize(faceRoiImage.data,ncnn::Mat::PIXEL_RGB, faceRoiImage.cols, faceRoiImage.rows,96,128);
    else
        ncnn_in = ncnn::Mat::from_pixels_resize(faceRoiImage.data,ncnn::Mat::PIXEL_RGB, faceRoiImage.cols, faceRoiImage.rows,192,256);

    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);
    ex_face.input("image",ncnn_in);
    ncnn::Mat out;
    ex_face.extract("save_infer_model/scale_0.tmp_1",out);

    ncnn::Mat argmax_out;
    std::vector<float> probs;
    argmax(out,argmax_out,probs);
    std::vector<float> coords = get_final_preds(out, argmax_out);

    keypoints.clear();
    for (int i = 0; i < coords.size() / 2; i++)
    {
        KeyPoint keypoint;
        keypoint.p = cv::Point(coords[i * 2]*w/ (float)out.w, coords[i * 2 + 1] * h / (float)out.h);
        keypoint.prob = probs[i];

        keypoints.push_back(keypoint);
    }

}
int NanoDet::draw(cv::Mat& rgb, const std::vector<KeyPoint>& keypoints)
{
    int skele_index[][2] = { {0,1},{0,2},{1,3},{2,4},{0,5},{0,6},{5,6},{5,7},{7,9},{6,8},{8,10},{11,12},
                             {5,11},{11,13},{13,15},{6,12},{12,14},{14,16} };
    int color_index[][3] = { {255, 0, 0},
                             {0, 0, 255},
                             {255, 0, 0},
                             {0, 0, 255},
                             {255, 0, 0},
                             {0, 0, 255},
                             {0, 255, 0},
                             {255, 0, 0},
                             {255, 0, 0},
                             {0, 0, 255},
                             {0, 0, 255},
                             {0, 255, 0},
                             {255, 0, 0},
                             {255, 0, 0},
                             {255, 0, 0},
                             {0, 0, 255},
                             {0, 0, 255},
                             {0, 0, 255}, };

    for (int i = 0; i < 18; i++)
    {
        if(keypoints[skele_index[i][0]].prob > 0.2 && keypoints[skele_index[i][1]].prob > 0.2)
            cv::line(rgb, keypoints[skele_index[i][0]].p, keypoints[skele_index[i][1]].p, cv::Scalar(color_index[i][0], color_index[i][1], color_index[i][2]), 2);
    }
    for (int i = 0; i < keypoints.size(); i++)
    {
        const KeyPoint& keypoint = keypoints[i];
        if (keypoint.prob > 0.2)
            cv::circle(rgb, keypoint.p, 3, cv::Scalar(100, 255, 150), -1);
    }

    return 0;
}
