#ifndef RF_CLASSIFIER_HPP_
#define RF_CLASSIFIER_HPP_

#include <Eigen/Core>
#include <vector>
#include <opencv2/ml/ml.hpp>        /* CvRTrees */
#include <mutex>                    /* std::mutex */

#include "base_classifier.hpp"
#include "common/types/feature.hpp"
#include "common/types/type.h"

namespace classifier {

    class RFClassifier : public BaseClassifier {

    public:
        explicit RFClassifier(const ClassifierParams& params);

        ~RFClassifier();

        virtual void classify(const Feature& cluster_feature,
                              ObjectType* classify_result,
                              double* probability);

        virtual void classify(const std::vector<Feature>& cluster_features,
                              std::vector<ObjectType>* classify_results,
                              std::vector<double>* probabilities);

        void train(const std::vector<Feature>& features,
                   const std::vector<Label>& labels);

        void retrain(const std::vector<Feature>& new_features,
                     const std::vector<Label>& new_labels);

        void test(const std::vector<Feature>& features,
                  const std::vector<Label>& labels,
                  std::vector<double>* probabilities = NULL);

        void load();

        void save() const;

        size_t size() const
        {
            return rtrees_.get_tree_count();
        }

        virtual std::string name() const
        {
            return "RFClassifier";
        }

    private:
        void classify(const CvRTrees& classifier,
                      const Feature& cluster_feature,
                      ObjectType* classify_result,
                      double* probability);

        void dummyClassify(const Feature& cluster_feature,
                           ObjectType* classify_result,
                           double* probability);

        void clone(const CvRTrees& rhs, CvRTrees& lhs);

    private:
        CvRTrees rtrees_new_;
        CvRTrees rtrees_;
        //Eigen::MatrixXd inverted_max_eigen_double_;
        //Eigen::MatrixXf inverted_max_eigen_float_;

        //samples pool statistics
        size_t num_samples_;
        size_t dim_feature_;

        //samples pool
        cv::Mat features_;
        cv::Mat labels_;
    }; /* RFClassifier */
}

#endif /* RF_CLASSIFIER_HPP_ */