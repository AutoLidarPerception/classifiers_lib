#include "classifiers/rf_classifier.hpp"

#include <ros/ros.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>        /* cv::Mat */

#include "common/time.hpp"              /* common::Clock */
#include "common/types/feature.hpp"     /* Feature */
#include "common/common.hpp"            /* common::displayPerformances */

namespace classifier {

    RFClassifier::RFClassifier(const ClassifierParams& params)
            : BaseClassifier(params)
    {
        if (!params.classifier_model_path.empty() && !params.rf_model_filename.empty()) {
            load();
        }

        /*inverted_max_eigen_double_.resize(1, 7);
        inverted_max_eigen_float_.resize(1, 7);
        for (int i = 0; i < 7; ++i) {
            inverted_max_eigen_double_(0, i) = 1.0
                                               / params.max_eigen_features_values[i];
            inverted_max_eigen_float_(0, i) = float(
                    1.0 / params.max_eigen_features_values[i]);
        }
        ROS_INFO_STREAM("inverted_max_eigen_double: " << inverted_max_eigen_double_);
        ROS_INFO_STREAM("inverted_max_eigen_float: " << inverted_max_eigen_float_);*/
    }

    RFClassifier::~RFClassifier()
    {
        if (params_.classifier_save) {
            save();
        }
    }

    /**
     * @brief basic classify function given classifier
     * @param classifier
     * @param cluster_feature
     * @param classify_result
     * @param probability
     */
    void RFClassifier::classify(const CvRTrees& classifier,
                                const Feature& cluster_feature,
                                ObjectType* classify_result,
                                double* probability)
    {
        cv::Mat opencv_sample(1, dim_feature_, CV_32FC1);
        for (size_t j = 0u; j < dim_feature_; ++j) {
            opencv_sample.at<float>(j) = cluster_feature.at(j).value;
        }
        //The function works for binary classification problems only, return [0,1]
        *probability = classifier.predict_prob(opencv_sample);

        if (*probability >= params_.rf_threshold_to_accept_object) {
            *classify_result = CARE;
        }
        else {
            *classify_result = DONTCARE;
        }
    }
    /**
     * @brief dummy classify as 50% probability
     * @param cluster_feature
     * @param classify_result
     * @param probability
     */
    void RFClassifier::dummyClassify(const Feature& cluster_feature,
                                     ObjectType* classify_result,
                                     double* probability)
    {
        (*classify_result) = CARE;
        (*probability) = 0.5;
    }
    /**
     * @brief one feature vector classify interface
     * @param cluster_feature
     * @param classify_result
     * @param probability
     */
    void RFClassifier::classify(const Feature& cluster_feature,
                                ObjectType* classify_result,
                                double* probability)
    {
        if (classify_result == nullptr || probability == nullptr) {
            return;
        }
        common::Clock clock;

        const size_t dim_feature = cluster_feature.size();
        assert(dim_feature == dim_feature_);

        if (trained_) {
            if (mtx_lock_.try_lock()) {
                if (retrained_) {
                    clone(rtrees_new_, rtrees_);
                    retrained_ = false;
                }
                mtx_lock_.unlock();
            }
            //otherwise, still use old classifier

            classify(rtrees_, cluster_feature, classify_result, probability);
        }
        else {
            dummyClassify(cluster_feature, classify_result, probability);
            ROS_WARN("[RFClassifier] Dummy classify without trained classifier.");
        }

        ROS_INFO_STREAM("Took " << clock.takeRealTime() << "ms to classify.");
    }
    /**
     * @brief several feature vectors classify interface
     * @param cluster_features
     * @param classify_results
     * @param probabilities
     */
    void RFClassifier::classify(const std::vector<Feature>& cluster_features,
                                std::vector<ObjectType>* classify_results,
                                std::vector<double>* probabilities)
    {
        if (classify_results == nullptr || probabilities == nullptr) {
            return;
        }

        common::Clock clock;
        const size_t num_samples = cluster_features.size();
        if (num_samples) {
            const size_t dim_feature = cluster_features[0].size();
            assert(dim_feature == dim_feature_);

            (*classify_results).clear();
            (*classify_results).resize(num_samples);
            (*probabilities).clear();
            (*probabilities).resize(num_samples);

            if (trained_) {
                if (mtx_lock_.try_lock()) {
                    if (retrained_) {
                        clone(rtrees_new_, rtrees_);
                        retrained_ = false;
                    }
                    mtx_lock_.unlock();
                }
                //otherwise, still use old classifier

                for (size_t f = 0u; f < num_samples; ++f) {
                    classify(rtrees_, cluster_features[f], &(*classify_results)[f], &(*probabilities)[f]);
                }
            }
            else {
                for (size_t f = 0u; f < num_samples; ++f) {
                    dummyClassify(cluster_features[f], &(*classify_results)[f], &(*probabilities)[f]);
                }
                ROS_WARN("[RFClassifier] Dummy classify without trained classifier.");
            }

        }

        ROS_INFO_STREAM("Took " << clock.takeRealTime() << "ms to classify.");
    }

    /**
     * @brief Generate features and labels for CvRTrees and train it, allow train repeatedly
     * @param features
     * @param labels
     */
    void RFClassifier::train(const std::vector<Feature>& features,
                             const std::vector<Label>& labels)
    {
        common::Clock clock;
        num_samples_ = features.size();
        dim_feature_ = features[0].size();
        ROS_INFO_STREAM("Training RF with " << num_samples_ << " of dimension "
                                            << dim_feature_ << ".");

        // init samples pool
        features_ = cv::Mat(params_.classifier_max_num_samples, dim_feature_, CV_32FC1);
        labels_ = cv::Mat(params_.classifier_max_num_samples, 1, CV_32FC1);

        for (size_t i = 0u; i < num_samples_; ++i) {
            for (size_t j = 0u; j < dim_feature_; ++j) {
                features_.at<float>(i, j) = features[i].at(j).value;
            }
            labels_.at<float>(i, 0) = labels[i];
        }

        float priors[] = {params_.rf_priors[0], params_.rf_priors[1]};

        // Random forest parameters.
        CvRTParams rtrees_params = CvRTParams(
                params_.rf_max_depth, params_.rf_min_sample_ratio * num_samples_,
                params_.rf_regression_accuracy, params_.rf_use_surrogates,
                params_.rf_max_categories, priors, params_.rf_calc_var_importance,
                params_.rf_n_active_vars, params_.rf_max_num_of_trees,
                params_.rf_accuracy,
                CV_TERMCRIT_EPS);

        rtrees_new_.train(features_.rowRange(0, num_samples_-1), CV_ROW_SAMPLE,
                          labels_.rowRange(0, num_samples_-1), cv::Mat(),
                          cv::Mat(), cv::Mat(), cv::Mat(), rtrees_params);

        ROS_INFO_STREAM("Tree count: " << rtrees_new_.get_tree_count() << ".");

        if (params_.rf_calc_var_importance) {
            cv::Mat variable_importance = rtrees_new_.getVarImportance();
            cv::Size variable_importance_size = variable_importance.size();
            if (variable_importance_size.height != 1.0) {
                ROS_WARN("Height of variable importance is not 1.");
            }
            if (variable_importance_size.width != dim_feature_) {
                ROS_WARN("Width of variable importance is the features dimension.");
            }

            // TODO Remove this cout (just for debugging).
            std::cout << "Variable importance: ";
            for (size_t i = 0u; i < dim_feature_; ++i) {
                std::cout << variable_importance.at<float>(i) << " ";
            }
            std::cout << std::endl;
        }

        clone(rtrees_new_, rtrees_);
        trained_ = true;

        // save trained model
        if (params_.classifier_save) {
            save();
        }

        ROS_INFO_STREAM("RF trained. Took " << clock.takeRealTime() << "ms.");
    }

    /**
     * @brief re-train RF adding new features and labels
     * @param features
     * @param labels
     */
    void RFClassifier::retrain(const std::vector<Feature>& new_features,
                               const std::vector<Label>& new_labels)
    {
        common::Clock clock;
        const size_t num_samples = new_features.size();
        const size_t dim_feature = new_features[0].size();

        assert(dim_feature == dim_feature_);

        ROS_INFO_STREAM("Re-training RF with " << num_samples << " of dimension "
                                               << dim_feature << ".");

        num_samples_ += num_samples;
        // safely check for fixed-size samples pool
        if (num_samples_ > params_.classifier_max_num_samples) {
            num_samples_ = params_.classifier_max_num_samples;

            //randomly insert
            std::vector<size_t> index(num_samples_ - num_samples);
            std::iota(index.begin(), index.end(), 0);
            std::random_shuffle(index.begin(), index.end());

            for (size_t i = index[0], new_i = 0u; new_i < num_samples; ++i, ++new_i) {
                for (size_t j = 0u; j < dim_feature; ++j) {
                    features_.at<float>(i, j) = new_features[new_i].at(j).value;
                }
                labels_.at<float>(i, 0) = new_labels[new_i];
            }
        }
        else {
            // append in reverse direction
            for (size_t i = num_samples_-1, new_i = 0u; new_i < num_samples; --i, ++new_i) {
                for (size_t j = 0u; j < dim_feature; ++j) {
                    features_.at<float>(i, j) = new_features[new_i].at(j).value;
                }
                labels_.at<float>(i, 0) = new_labels[new_i];
            }
        }

        thread_pool_->Enqueue([this]{
            common::Clock clock;
            const size_t num_samples = num_samples_;
            // Random forest parameters
            float priors[] = {params_.rf_priors[0], params_.rf_priors[1]};
            CvRTParams rtrees_params = CvRTParams(
                    params_.rf_max_depth, params_.rf_min_sample_ratio * num_samples,
                    params_.rf_regression_accuracy, params_.rf_use_surrogates,
                    params_.rf_max_categories, priors, params_.rf_calc_var_importance,
                    params_.rf_n_active_vars, params_.rf_max_num_of_trees,
                    params_.rf_accuracy,
                    CV_TERMCRIT_EPS);

            //lock when re-training a new classifier
            mtx_lock_.lock();
            rtrees_new_.train(features_.rowRange(0, num_samples-1), CV_ROW_SAMPLE,
                              labels_.rowRange(0, num_samples-1), cv::Mat(),
                              cv::Mat(), cv::Mat(), cv::Mat(), rtrees_params);
            retrained_ = true;
            mtx_lock_.unlock();

            ROS_INFO_STREAM("Tree count: " << rtrees_new_.get_tree_count() << ".");

            if (params_.rf_calc_var_importance) {
                cv::Mat variable_importance = rtrees_new_.getVarImportance();
                cv::Size variable_importance_size = variable_importance.size();
                if (variable_importance_size.height != 1.0) {
                    ROS_WARN("Height of variable importance is not 1.");
                }
                if (variable_importance_size.width != dim_feature_) {
                    ROS_WARN("Width of variable importance is the features dimension.");
                }

                //TODO Remove this cout (just for debugging).
                std::cout << "Variable importance: ";
                for (size_t i = 0u; i < dim_feature_; ++i) {
                    std::cout << variable_importance.at<float>(i) << " ";
                }
                std::cout << std::endl;
            }

            ROS_WARN("[thread pool] RF retrained, %d samples. Took %lf ms.", num_samples, clock.takeRealTime());
        });

        ROS_INFO("RF retrained, %u samples. Took %lf ms.", num_samples_, clock.takeRealTime());
    }

    /**
     * @brief Test trained RF
     * @param features
     * @param labels
     * @param probabilities
     */
    void RFClassifier::test(const std::vector<Feature>& features,
                            const std::vector<Label>& labels,
                            std::vector<double>* probabilities)
    {
        common::Clock clock;
        const unsigned int num_samples = features.size();
        const unsigned int dim_feature = features[0].size();
        ROS_INFO_STREAM("Testing RF with " << num_samples << " samples of dimension "
                                           << dim_feature << ".");

        if (probabilities != NULL) {
            (*probabilities).resize(num_samples, 1);
        }

        if (num_samples > 0u) {
            unsigned int tp = 0u, fp = 0u, tn = 0u, fn = 0u;
            for (unsigned int i = 0u; i < num_samples; ++i) {
                cv::Mat opencv_sample(1, dim_feature, CV_32FC1);
                for (unsigned int j = 0u; j < dim_feature; ++j) {
                    opencv_sample.at<float>(j) = features[i].at(j).value;
                }
                //The function works for binary classification problems only, return [0,1]
                double probability = rtrees_.predict_prob(opencv_sample);
                if (probability >= params_.rf_threshold_to_accept_object) {
                    if (labels[i] == 1.0) {
                        ++tp;
                    } else {
                        ++fp;
                    }
                } else {
                    if (labels[i] == 0.0) {
                        ++tn;
                    } else {
                        ++fn;
                    }
                }
                if (probabilities != NULL) {
                    (*probabilities)[i] = probability;
                }
            }
            common::displayPerformances(tp, tn, fp, fn);
        }

        ROS_INFO_STREAM("Took " << clock.takeRealTime() << "ms to test.");
    }

    /**
     * @brief RF model save to model file and load from model file
     * @param filename
     */
    void RFClassifier::load()
    {
        if (params_.classifier_model_path.empty()){
            ROS_ERROR("Failed to load Random Forest model: not specified model path!");
            return;
        }
        if (params_.rf_model_filename.empty()){
            ROS_ERROR("Failed to load Random Forest model: not specified model name!");
            return;
        }
        std::string model_file = params_.classifier_model_path + "/" + params_.rf_model_filename;
        rtrees_.load(model_file.c_str());
        ROS_INFO("Loaded Random Forest classifier from: %s.", model_file.c_str());

        trained_ = true;

        dim_feature_ = rtrees_.getVarImportance().cols;
    }
    void RFClassifier::save() const
    {
        if (trained_) {
            std::string save_name = params_.classifier_model_path + "/";
            save_name += "random_forest_" + std::to_string(rtrees_.get_tree_count()) + "trees_";
            save_name += common::getCurrentTimestampString();
            save_name += ".xml";

            rtrees_.save(save_name.c_str());
            ROS_INFO_STREAM("Saved the Random Forest classifier to: " << save_name << ".");
        }
        else {
            ROS_WARN("An empty classifier, no need to save.");
        }
    }

    void RFClassifier::clone(const CvRTrees& rhs, CvRTrees& lhs)
    {
        std::string config_file = params_.classifier_model_path + "/" + config_name_;
        rhs.save(config_file.c_str());

        lhs.load(config_file.c_str());
    }
}