#ifndef SVM_CLASSIFIER_HPP_
#define SVM_CLASSIFIER_HPP_

#include "base_classifier.hpp"
#include "common/types/feature.hpp"
#include "libsvm/svm.h"

namespace classifier {

    struct svm_feature_node {
        svm_feature_node(size_t dimesion)
                :size_(dimesion)
        {
            // 1 more size for end index (-1)
            feature_node_ = (struct svm_node*) malloc((dimesion + 1) * sizeof(struct svm_node));
        }

        struct svm_node& operator[](size_t idx) const
        {
            return feature_node_[idx];
        }

        struct svm_node* get() const {
            return feature_node_;
        }

        size_t size_;
        struct svm_node* feature_node_;
    };

    struct svm_feature_range {
        double min;
        double max;
    };

    class SVMClassifier : public BaseClassifier {
        typedef struct svm_node* svm_feature_node_t;

    public:
        SVMClassifier(ClassifierParams params);

        ~SVMClassifier();

        void classify(const Feature& cluster_feature,
                      ObjectType* classify_result,
                      double* probability);

        void classify(const std::vector<Feature>& cluster_features,
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
            return svm_model_->l;
        }

        virtual std::string name() const
        {
            return "SVMClassifier";
        }

    private:
        void classify(const struct svm_model* classifier,
                      const Feature& cluster_feature,
                      ObjectType* classify_result,
                      double* probability);
        void dummyClassify(const Feature& cluster_feature,
                           ObjectType* classify_result,
                           double* probability);

        void feature2SvmNode(const Feature& feature, struct svm_feature_node* svm_feature_node);

        void saveSvmRange(const std::string& range_file_name) const;

        //feature values normalize to svm features' range
        double normalizeFeatureValue(const double& value, const size_t& feature_index) const;

        //find the best SVM training parameters
        void findBestTrainingParams();

        //deep copy of svm_model
        svm_model* clone(const svm_model* rhs);

    private:
        //limit SVM's nodes
        //size_t svm_node_size_;
        size_t svm_feature_size_;

        ///@note svm range for each dimension of feature
        std::vector<struct svm_feature_range> svm_feature_xrange_;

        /**
         * svm_train(&svm_problem_, &svm_parameter_)
         * svm_problem_: defined by losts of [x,y] samples:x-->features,y-->label(1;-1)
         */
        struct svm_parameter svm_parameter_;
        // samples pool
        struct svm_problem svm_problem_;

        //SVM model(load then train)
        struct svm_model* svm_model_;
        struct svm_model* svm_model_new_;

        //bool use_svm_model_;
        bool is_probability_model_;
    };
}

#endif