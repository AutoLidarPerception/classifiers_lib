#ifndef CLASSIFIER_MANAGER_HPP_
#define CLASSIFIER_MANAGER_HPP_

#include "base_classifier.hpp"
#include "rf_classifier.hpp"
#include "svm_classifier.hpp"

#include <ros/ros.h>

namespace classifier {

    static std::unique_ptr<BaseClassifier> createClassifier(ClassifierParams params)
    {
        std::unique_ptr<BaseClassifier> classifier;
        if (params.classifier_type == "RF") {
            classifier = std::unique_ptr<BaseClassifier>(new RFClassifier(params));
            ROS_INFO("[learning] Instance of Random Forest Classifier created.");
        }
        else if (params.classifier_type == "SVM") {
            classifier = std::unique_ptr<BaseClassifier>(new SVMClassifier(params));
            ROS_INFO("[learning] Instance of SVM Classifier created.");
        }
        else {
            classifier = std::unique_ptr<BaseClassifier>(new RFClassifier(params));
            ROS_INFO("[learning] Instance of Random Forest Classifier created.");
        }
        return classifier;
    }
}

#endif /* CLASSIFIER_MANAGER_HPP_ */