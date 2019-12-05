// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>

#include "core.hpp"


namespace {
cv::Rect TruncateToValidRect(const cv::Rect& rect,
                             const cv::Size& size) {
    auto tl = rect.tl(), br = rect.br();
    tl.x = std::max(0, std::min(size.width - 1, tl.x));
    tl.y = std::max(0, std::min(size.height - 1, tl.y));
    br.x = std::max(0, std::min(size.width, br.x));
    br.y = std::max(0, std::min(size.height, br.y));
    int w = std::max(0, br.x - tl.x);
    int h = std::max(0, br.y - tl.y);
    return cv::Rect(tl.x, tl.y, w, h);
}

cv::Rect IncreaseRect(const cv::Rect& r, float coeff_x,
                      float coeff_y)  {
    cv::Point2f tl = r.tl();
    cv::Point2f br = r.br();
    cv::Point2f c = (tl * 0.5f) + (br * 0.5f);
    cv::Point2f diff = c - tl;
    cv::Point2f new_diff{diff.x * coeff_x, diff.y * coeff_y};
    cv::Point2f new_tl = c - new_diff;
    cv::Point2f new_br = c + new_diff;

    cv::Point new_tl_int {static_cast<int>(std::floor(new_tl.x)), static_cast<int>(std::floor(new_tl.y))};
    cv::Point new_br_int {static_cast<int>(std::ceil(new_br.x)), static_cast<int>(std::ceil(new_br.y))};

    return cv::Rect(new_tl_int, new_br_int);
}
}  // namespace


struct DetectorConfig {
    explicit DetectorConfig(const std::string& path_to_model,
                            const std::string& path_to_weights)
        : path_to_model(path_to_model), path_to_weights(path_to_weights) {}

    /** @brief Path to model description */
    std::string path_to_model;
    /** @brief Path to model weights */
    std::string path_to_weights;
    float confidence_threshold{0.5f};
    float increase_scale_x{1.f};
    float increase_scale_y{1.f};
    bool  is_async = false;
};

class Detector {

public:

    static constexpr float SSD_EMPTY_DETECTIONS_INDICATOR = -1.0;

    Detector() = default;

    Detector(const DetectorConfig&                     config,
             const InferenceEngine::Core&              ie,
             const std::string            	       deviceName,
             const bool                                autoResize,
             const std::map<std::string, std::string>& pluginConfig,
	     const cv::Size&                           new_input_resolution,
	     unsigned&                                 nireq) :
             config_(config),
    	     ie_(ie) {

    	InferenceEngine::CNNNetReader net_reader;
    	net_reader.ReadNetwork(config.path_to_model);
    	net_reader.ReadWeights(config.path_to_weights);
    	if (!net_reader.isParseSuccess()) {
            THROW_IE_EXCEPTION << "Cannot load model";
    	}
        
        InferenceEngine::CNNNetwork network = net_reader.getNetwork();
    	InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    	if (1 == inputInfo.size() || 2 == inputInfo.size()) {
            for (const std::pair<std::string, InferenceEngine::InputInfo::Ptr>& input : inputInfo) {
                InferenceEngine::InputInfo::Ptr inputInfo = input.second;
                if (4 == inputInfo->getTensorDesc().getDims().size()) {
                    inputInfo->setPrecision(InferenceEngine::Precision::U8);
                    inputInfo->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
                    input_name_ = input.first;
                } else if (InferenceEngine::SizeVector{1, 6} == inputInfo->getTensorDesc().getDims()) {
                    inputInfo->setPrecision(InferenceEngine::Precision::FP32);
                    im_info_name_ = input.first;
                } else {
                    THROW_IE_EXCEPTION << "Unknown input for Person Detection network";
                }
            }
            if (input_name_.empty()) {
                THROW_IE_EXCEPTION << "No image input for Person Detection network found";
            }
        } else {
            THROW_IE_EXCEPTION << "Person Detection network should have one or two inputs";
        }

        InferenceEngine::InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
        if (autoResize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
        }

        InferenceEngine::SizeVector input_dims = inputInfoFirst->getInputData()->getTensorDesc().getDims();
        input_dims[2]         = static_cast<size_t>(new_input_resolution.height);
        input_dims[3]         = static_cast<size_t>(new_input_resolution.width);

        std::map<std::string, InferenceEngine::SizeVector> input_shapes;
        input_shapes[network.getInputsInfo().begin()->first] = input_dims;
        network.reshape(input_shapes);

        InferenceEngine::OutputsDataMap outputInfo(net_reader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            THROW_IE_EXCEPTION << "Person Detection network should have only one output";
        }
        InferenceEngine::DataPtr& _output = outputInfo.begin()->second;
        output_name_ = outputInfo.begin()->first;

        const InferenceEngine::CNNLayerPtr outputLayer = net_reader.getNetwork().getLayerByName(output_name_.c_str());
        if (outputLayer->type != "DetectionOutput") {
            THROW_IE_EXCEPTION << "Person Detection network output layer(" + outputLayer->name +
                ") should be DetectionOutput, but was " +  outputLayer->type;
        }

        if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
            THROW_IE_EXCEPTION << "Person Detection network output layer (" +
                output_name_ + ") should have num_classes integer attribute";
        }

        const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
    	if (outputDims.size() != 4) {
            THROW_IE_EXCEPTION << "Person Detection network output dimensions not compatible shoulld be 4, but was " +
            	std::to_string(outputDims.size());
    	}
    	max_detections_count_ = outputDims[2];
    	object_size_ = outputDims[3];
    	if (object_size_ != 7) {
            THROW_IE_EXCEPTION << "Person Detection network output layer should have 7 as a last dimension";
        }
    	_output->setPrecision(InferenceEngine::Precision::FP32);
    	_output->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(_output->getDims()));

    	net_ = ie_.LoadNetwork(net_reader.getNetwork(), deviceName, pluginConfig);
	nireq = net_.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    }

    InferenceEngine::InferRequest createInferRequest() {
        return net_.CreateInferRequest();
    }

    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& frame) {

	width_  = static_cast<float>(frame.cols);
	height_ = static_cast<float>(frame.rows);

	InferenceEngine::Blob::Ptr inputBlob = inferRequest.GetBlob(input_name_);

        if (InferenceEngine::Layout::NHWC == inputBlob->getTensorDesc().getLayout()) {  // autoResize is set
            if (!frame.isSubmatrix()) {
                // just wrap Mat object with Blob::Ptr without additional memory allocation
                InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(frame);
                inferRequest.SetBlob(input_name_, frameBlob);
            } else {
                throw std::logic_error("Sparse matrix are not supported");
            }
        } else {
	    matU8ToBlob<uint8_t>(frame, inputBlob);
	}

	if (!im_info_name_.empty()) {
	    float* buffer = inferRequest.GetBlob(im_info_name_)->buffer().as<float*>();
	    buffer[0] = static_cast<float>(inputBlob->getTensorDesc().getDims()[2]);
	    buffer[1] = static_cast<float>(inputBlob->getTensorDesc().getDims()[3]);
	    buffer[2] = buffer[4] = static_cast<float>(inputBlob->getTensorDesc().getDims()[3]) / width_;
	    buffer[3] = buffer[5] = static_cast<float>(inputBlob->getTensorDesc().getDims()[2]) / height_;
	}
    }

    TrackedObjects getResults(InferenceEngine::InferRequest& inferRequest,
			      cv::Size                       upscale,
                              std::ostream*                  rawResults = nullptr) {

        // there is no big difference if InferReq of detector from another device is passed because the processing is the same for the same topology
        TrackedObjects results;
        const float *data = inferRequest.GetBlob(output_name_)->buffer().as<float *>();

	for (int det_id = 0; det_id < max_detections_count_; ++det_id) {
	    const int start_pos = det_id * object_size_;

	    const float batchID = data[start_pos];
	    if (batchID == SSD_EMPTY_DETECTIONS_INDICATOR) {
		break;
	    }

	    const float score = std::min(std::max(0.0f, data[start_pos + 2]), 1.0f);
	    const float x0    = std::min(std::max(0.0f, data[start_pos + 3]), 1.0f) * upscale.width;
	    const float y0    = std::min(std::max(0.0f, data[start_pos + 4]), 1.0f) * upscale.height;
            const float x1    = std::min(std::max(0.0f, data[start_pos + 5]), 1.0f) * upscale.width;
            const float y1    = std::min(std::max(0.0f, data[start_pos + 6]), 1.0f) * upscale.height;

            TrackedObject object;
	    object.confidence = score;
	    object.rect = cv::Rect(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
		                             static_cast<int>(round(static_cast<double>(y0)))),
		                   cv::Point(static_cast<int>(round(static_cast<double>(x1))),
		                             static_cast<int>(round(static_cast<double>(y1)))));

	    object.rect = TruncateToValidRect(IncreaseRect(object.rect,
		                                           config_.increase_scale_x,
		                                           config_.increase_scale_y),
		                              cv::Size(static_cast<int>(upscale.width), static_cast<int>(upscale.height)));

	    if (object.confidence > config_.confidence_threshold && object.rect.area() > 0) {
		results.emplace_back(object);
	    }
	    if (rawResults) {
                *rawResults << "[" << det_id << "] element, prob = " << score <<
                               "    (" << object.rect.x << "," << object.rect.y <<
                               ")-(" << object.rect.width << "," << object.rect.height << ")" << std::endl;
            }
	}
        return results;
    }

private:

    DetectorConfig config_;
    InferenceEngine::Core ie_;  // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork

    InferenceEngine::ExecutableNetwork net_;
    std::string input_name_;
    std::string output_name_;
    std::string im_info_name_;
    int max_detections_count_;
    int object_size_;

    float width_  = 0;
    float height_ = 0;
};

