#ifndef BASE_CLASSIFIER_HPP_
#define BASE_CLASSIFIER_HPP_

#include <string>
#include <vector>
#include <mutex>                        /* std::mutex */

#include "common/types/feature.hpp"             /* Feature */
#include "common/types/type.h"                  /* ObjectType */
#include "common/thread_pool.hpp"               /* common::ThreadPool */

namespace classifier {

    class BaseClassifier {
    public:
        BaseClassifier(ClassifierParams params)
                : params_(params),
                  trained_(false),
                  retrained_(false)
        {
            thread_pool_ = std::unique_ptr<common::ThreadPool>(new common::ThreadPool(1));
            config_name_ = ".config.model";
        }

        virtual void classify(const Feature& cluster_feature,
                              ObjectType* classify_result,
                              double* probability) = 0;

        virtual void classify(const std::vector<Feature>& cluster_features,
                              std::vector<ObjectType>* classify_results,
                              std::vector<double>* probabilities) = 0;


        virtual void train(const std::vector<Feature>& features,
                           const std::vector<Label>& labels) = 0;

        virtual void retrain(const std::vector<Feature>& new_features,
                             const std::vector<Label>& new_labels) = 0;

        virtual void test(const std::vector<Feature>& features,
                          const std::vector<Label>& labels,
                          std::vector<double>* probabilities = NULL) = 0;

        /**
         * @brief load and save model specified by Classifier parameters
         */
        virtual void load() = 0;
        virtual void save() const = 0;

        virtual size_t size() const = 0;

        virtual bool trained() const
        {
            return trained_;
        }

        virtual std::string name() const = 0;

    protected:
        bool trained_;
        bool retrained_;

        //temporary file used for deep copy of classifier
        std::string config_name_;

        //TODO "classifier"&"retrained_" protect lock, new re-trained classifier and current using classifier
        std::mutex mtx_lock_;
        //training process thread
        boost::shared_ptr<common::ThreadPool> thread_pool_;

        ClassifierParams params_;
    }; /* class BaseClassifier */
}

#endif /* BASE_CLASSIFIER_HPP_ */