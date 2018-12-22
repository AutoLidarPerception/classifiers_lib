#include "classifiers/svm_classifier.hpp"

#include <ros/ros.h>
#include <algorithm>                        /* std::min/std::max */
#include <fstream>
#include <boost/algorithm/string.hpp>       /* boost::split */

#include "common/time.hpp"                  /* common::Clock */
#include "common/types/feature.hpp"         /* Feature */
#include "common/common.hpp"                /* common::displayPerformances */

namespace classifier {
    SVMClassifier::SVMClassifier(ClassifierParams params)
            : BaseClassifier(params)
    {
        if (!params.classifier_model_path.empty()
            && !params.svm_model_filename.empty() && !params.svm_range_filename.empty()) {
            load();
        }

        //SVM parameters from online_learning
        svm_parameter_.svm_type = C_SVC; // default C_SVC
        svm_parameter_.kernel_type = RBF; // default RBF
        svm_parameter_.degree = 3; // default 3
        svm_parameter_.gamma = 0.02; // default 1.0/(float)FEATURE_SIZE
        svm_parameter_.coef0 = 0; // default 0
        svm_parameter_.cache_size = 256; // default 100
        svm_parameter_.eps = 0.001; // default 0.001
        svm_parameter_.C = 8; // default 1
        svm_parameter_.nr_weight = 0;
        svm_parameter_.weight_label = NULL;
        svm_parameter_.weight = NULL;
        svm_parameter_.nu = 0.5;
        svm_parameter_.p = 0.1;
        svm_parameter_.shrinking = 0;
        svm_parameter_.probability = 1;

        //svm_node_size_ = (round_positives_ + round_negatives_) * (max_trains_ - 1) + init_positives_ + init_negatives_;
        //svm_node_size_ = 0;
        //current samples' number
        svm_problem_.l = 0;
        //svm_node_size_ samples' label
        svm_problem_.y = (double*) malloc(params_.classifier_max_num_samples * sizeof(double));
        //svm_node_size_ samples' features
        svm_problem_.x = (struct svm_node**) malloc(params_.classifier_max_num_samples * sizeof(struct svm_node*));
    }

    SVMClassifier::~SVMClassifier()
    {
        if (params_.classifier_save) {
            save();
        }

        svm_free_and_destroy_model(&svm_model_);
        svm_free_and_destroy_model(&svm_model_new_);
        //free(svm_node_);
        svm_destroy_param(&svm_parameter_);
        free(svm_problem_.y);
        for (size_t node = 0u; node < params_.classifier_max_num_samples; ++node) {
            free(svm_problem_.x[node]);
        }
        free(svm_problem_.x);
    }

    /**
     * @brief basic classify function given classifier
     * @param classifier
     * @param cluster_feature
     * @param classify_result
     * @param probability
     */
    void SVMClassifier::classify(const struct svm_model* classifier,
                                 const Feature& cluster_feature,
                                 ObjectType* classify_result,
                                 double* probability)
    {
        //--------- Construct svm feature node
        struct svm_feature_node feature_node(svm_feature_size_+1);
        feature2SvmNode(cluster_feature, &feature_node);
        //--------- Normalize feature value to [-1, 1]
        for (size_t idx = 0u; idx < svm_feature_size_; ++idx) {
            feature_node[idx].value = normalizeFeatureValue(feature_node[idx].value, idx);
        }

        //--------- Predict
        ///TODO 检查模式是否包含做概率估计时所需的信息(可能是二分类甚至多分类)
        if (is_probability_model_) {
            double prob_estimates[classifier->nr_class];
            svm_predict_probability(classifier, feature_node.get(), prob_estimates);
            if (prob_estimates[0] > params_.svm_threshold_to_accept_object) {
                *classify_result = CARE;
            }
            else {
                *classify_result = DONTCARE;
            }
            *probability = prob_estimates[0];
        } else {
            if (svm_predict(classifier, feature_node.get()) == 1) {
                *classify_result = CARE;
                *probability = 1.;
            }
            else {
                *classify_result = DONTCARE;
                *probability = 0.;
            }
        }
    }
    /**
     * @brief dummy classify as 50% probability
     * @param cluster_feature
     * @param classify_result
     * @param probability
     */
    void SVMClassifier::dummyClassify(const Feature& cluster_feature,
                                      ObjectType* classify_result,
                                      double* probability)
    {
        (*classify_result) = CARE;
        (*probability) = 0.5;
    }

    void SVMClassifier::classify(const Feature& cluster_feature,
                                 ObjectType* classify_result,
                                 double* probability)
    {
        if (classify_result == nullptr || probability == nullptr) {
            return;
        }

        common::Clock clock;

        const size_t dim_feature = cluster_feature.size();
        if (dim_feature != svm_feature_size_) {
            ROS_ERROR("Inconsistent in feature size, svm model trained with %u dimension feature, "
                              "while classify feature is %u dimension.", svm_feature_size_, dim_feature);
            classify_result = NULL;
            probability = NULL;
            return;
        }
        assert(dim_feature == svm_feature_size_);

        if (trained_) {
            if (mtx_lock_.try_lock()) {
                if (retrained_) {
                    svm_model_ = clone(svm_model_new_);
                    is_probability_model_ = svm_check_probability_model(svm_model_) ? true : false;
                    retrained_ = false;
                }
                mtx_lock_.unlock();
            }
            //otherwise, still use old classifier

            classify(svm_model_, cluster_feature, classify_result, probability);
        }
        else {
            dummyClassify(cluster_feature, classify_result, probability);
            ROS_WARN("[SVMClassifier] Dummy classify without trained classifier.");
        }

        ROS_INFO_STREAM("Took " << clock.takeRealTime() << "ms to classify.");
    }

    void SVMClassifier::classify(const std::vector<Feature>& cluster_features,
                                 std::vector<ObjectType>* classify_results,
                                 std::vector<double>* probabilities)
    {
        if (classify_results == nullptr || probabilities == nullptr) {
            return;
        }

        common::Clock clock;
        (*classify_results).clear();
        (*probabilities).clear();

        const size_t num_samples = cluster_features.size();

        if (num_samples) {
            const size_t dim_feature = cluster_features[0].size();
            if (dim_feature != svm_feature_size_) {
                ROS_ERROR("Inconsistent in feature size, svm model trained with %u dimension feature, "
                                  "while classify feature is %u dimension.", svm_feature_size_, dim_feature);
                classify_results = NULL;
                probabilities = NULL;
                return;
            }
            assert(dim_feature == svm_feature_size_);

            (*classify_results).resize(num_samples);
            (*probabilities).resize(num_samples);

            if (trained_) {
                if (mtx_lock_.try_lock()) {
                    if (retrained_) {
                        svm_model_ = clone(svm_model_new_);
                        is_probability_model_ = svm_check_probability_model(svm_model_) ? true : false;
                        retrained_ = false;
                    }
                    mtx_lock_.unlock();
                }
                //otherwise, still use old classifier

                for (size_t f = 0u; f < cluster_features.size(); ++f) {
                    classify(svm_model_, cluster_features[f], &(*classify_results)[f], &(*probabilities)[f]);
                }
            }
            else {
                for (size_t f = 0u; f < cluster_features.size(); ++f) {
                    dummyClassify(cluster_features[f], &(*classify_results)[f], &(*probabilities)[f]);
                }
                ROS_WARN("[SVMClassifier] Dummy classify without trained classifier.");
            }
        }

        ROS_INFO_STREAM("Took " << clock.takeRealTime() << "ms to classify.");
    }

    void SVMClassifier::train(const std::vector<Feature>& features,
                              const std::vector<Label>& labels)
    {
        common::Clock clock;

        //check samples pool allocated well
        if (svm_problem_.x == nullptr || svm_problem_.y == nullptr) {
            ROS_ERROR("Failed to check samples pool allocated well.");
            return;
        }

        svm_problem_.l = features.size();
        svm_feature_size_ = features[0].size();
        // Finish allocating samples pool
        for (size_t i = 0u; i < params_.classifier_max_num_samples; ++i) {
            //1 more for end index (-1)
            svm_problem_.x[i] = (struct svm_node *) malloc((svm_feature_size_ + 1) * sizeof(struct svm_node));
        }

        ROS_INFO_STREAM("Training SVM with " << svm_problem_.l << " of dimension "
                                             << svm_feature_size_ << ".");

        //------------ Scale the current data, find svm range for each dimension of feature
        svm_feature_xrange_.resize(svm_feature_size_);
        for (size_t idx = 0u; idx < svm_feature_size_; ++idx) {
            svm_feature_xrange_[idx].min = svm_feature_xrange_[idx].max = features[0].at(idx).value;
        }
        for (size_t node = 1u; node < svm_problem_.l; ++node) {
            for (size_t idx = 0u; idx < svm_feature_size_; ++idx) {
                svm_feature_xrange_[idx].min = std::min(svm_feature_xrange_[idx].min, features[node].at(idx).value);
                svm_feature_xrange_[idx].max = std::max(svm_feature_xrange_[idx].max, features[node].at(idx).value);
            }
        }

        //TODO construct svm problem
        for (size_t node = 0u; node < svm_problem_.l; ++node) {
            for (size_t idx = 0u; idx < svm_feature_size_; ++idx) {
                svm_problem_.x[node][idx].index = idx + 1;

                svm_problem_.x[node][idx].value = normalizeFeatureValue(features[node].at(idx).value, idx);
            }
            //TODO Important!!! end index (-1)
            svm_problem_.x[node][svm_feature_size_].index = -1;

            svm_problem_.y[node] = labels[node];
        }
        //------------- train
        if (params_.svm_find_the_best_training_parameters) {
            findBestTrainingParams();
        }
        svm_model_new_ = svm_train(&svm_problem_, &svm_parameter_);

        ROS_INFO_STREAM("Training finished. Took " << clock.takeRealTime() << "ms.");

        svm_model_ = clone(svm_model_new_);
        is_probability_model_ = svm_check_probability_model(svm_model_) ? true : false;
        trained_ = true;

        if (params_.classifier_save) {
            save();
        }
    }

    void SVMClassifier::retrain(const std::vector<Feature>& new_features,
                                const std::vector<Label>& new_labels)
    {
        common::Clock clock;
        const size_t num_samples = new_features.size();
        const size_t dim_feature = new_features[0].size();

        assert(dim_feature == svm_feature_size_);

        ROS_INFO_STREAM("Re-training SVM with " << num_samples << " of dimension "
                                                << dim_feature << ".");

        //------------ Update svm range for each dimension of feature
        for (size_t node = 0u; node < num_samples; ++node) {
            for (size_t idx = 0u; idx < svm_feature_size_; ++idx) {
                svm_feature_xrange_[idx].min = std::min(svm_feature_xrange_[idx].min, new_features[node].at(idx).value);
                svm_feature_xrange_[idx].max = std::max(svm_feature_xrange_[idx].max, new_features[node].at(idx).value);
            }
        }

        svm_problem_.l += num_samples;
        // safely check for fixed-size samples pool
        if (svm_problem_.l > params_.classifier_max_num_samples) {
            svm_problem_.l = params_.classifier_max_num_samples;

            //randomly insert
            std::vector<size_t> index(svm_problem_.l - num_samples);
            std::iota(index.begin(), index.end(), 0);
            std::random_shuffle(index.begin(), index.end());

            for (size_t node = index[0], new_i = 0u; new_i < num_samples; ++node, ++new_i) {
                for (size_t idx = 0u; idx < svm_feature_size_; ++idx) {
                    svm_problem_.x[node][idx].index = idx + 1;

                    svm_problem_.x[node][idx].value = normalizeFeatureValue(new_features[new_i].at(idx).value, idx);
                }
                //TODO Important!!! end index (-1)
                svm_problem_.x[node][svm_feature_size_].index = -1;

                svm_problem_.y[node] = new_labels[new_i];
            }
        }
        else {
            // append in reverse direction
            for (size_t node = svm_problem_.l-1, new_i = 0u; new_i < num_samples; --node, ++new_i) {
                for (size_t idx = 0u; idx < svm_feature_size_; ++idx) {
                    svm_problem_.x[node][idx].index = idx + 1;

                    svm_problem_.x[node][idx].value = normalizeFeatureValue(new_features[new_i].at(idx).value, idx);
                }
                //TODO Important!!! end index (-1)
                svm_problem_.x[node][svm_feature_size_].index = -1;

                svm_problem_.y[node] = new_labels[new_i];
            }
        }
        //------------- train
        thread_pool_->Enqueue([this]{
            if (params_.svm_find_the_best_training_parameters) {
                findBestTrainingParams();
            }

            common::Clock clock;
            //lock when re-training a new classifier
            mtx_lock_.lock();
            svm_model_new_ = svm_train(&svm_problem_, &svm_parameter_);
            retrained_ = true;
            mtx_lock_.unlock();
            ROS_WARN("[thread pool] SVM retrained, %d samples. Took %lf ms.", svm_problem_.l, clock.takeRealTime());
        });

        ROS_INFO("SVM retrained, %d samples. Took %lf ms.", svm_problem_.l, clock.takeRealTime());
    }

    void SVMClassifier::test(const std::vector<Feature>& features,
                             const std::vector<Label>& labels,
                             std::vector<double>* probabilities)
    {
        common::Clock clock;
        const unsigned int num_samples = features.size();
        const unsigned int dim_feature = features[0].size();
        ROS_INFO_STREAM("Testing SVM with " << num_samples << " samples of dimension "
                                            << dim_feature << ".");

        if (probabilities != NULL) {
            (*probabilities).resize(num_samples, 1);
        }

        if (num_samples > 0u) {
            unsigned int tp = 0u, fp = 0u, tn = 0u, fn = 0u;
            for (unsigned int i = 0u; i < num_samples; ++i) {
                ObjectType classify_result;
                double probability;
                classify(features.at(i), &classify_result, &probability);

                if (classify_result == CARE) {
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

    void SVMClassifier::load()
    {
        if (params_.classifier_model_path.empty()){
            ROS_ERROR_STREAM("Failed to load SVM model: not specified model path!");
            return;
        }
        if (params_.svm_model_filename.empty()){
            ROS_ERROR_STREAM("Failed to load SVM model: not specified model file name!");
            return;
        }
        if (params_.svm_range_filename.empty()){
            ROS_ERROR_STREAM("Failed to load SVM model: not specified range file name!");
            return;
        }

        const std::string& model_name = params_.classifier_model_path + "/" + params_.svm_model_filename;
        const std::string& range_name = params_.classifier_model_path + "/" + params_.svm_range_filename;

        if ((svm_model_ = svm_load_model(model_name.c_str())) == NULL) {
            ROS_ERROR("Can not load SVM model: '%s'!", model_name.c_str());
            return;
        } else {
            std::fstream range_file;
            range_file.open(range_name.c_str(), std::fstream::in);
            if (!range_file.is_open()) {
                ROS_ERROR("Can not load range file: '%s'!", range_name.c_str());
                return;
            } else {
                ROS_INFO("Load SVM model from '%s'.", model_name.c_str());
                is_probability_model_ = svm_check_probability_model(svm_model_) ? true : false;

                // load range file, for more details: https://github.com/cjlin1/libsvm/
                ROS_INFO("Load SVM range from '%s'.", range_name.c_str());
                std::string line;
                std::vector<std::string> params;
                //skip range name "x"
                std::getline(range_file, line);
                //load xrange lower/upper
                std::getline(range_file, line);
                boost::split(params, line, boost::is_any_of(" "));
                params_.svm_feature_range_lower = atof(params[0].c_str());
                params_.svm_feature_range_upper = atof(params[1].c_str());
                //load xrange for each feature
                //int i = 0;
                while (std::getline(range_file, line)) {
                    boost::split(params, line, boost::is_any_of(" "));
                    struct svm_feature_range xrange;
                    xrange.min = atof(params[1].c_str());
                    xrange.max = atof(params[2].c_str());
                    svm_feature_xrange_.push_back(xrange);
                    //i++;
                    // std::cerr << i << " " <<  svm_scale_range_[i][0] << " " << svm_scale_range_[i][1] << std::endl;
                }

                svm_feature_size_ = svm_feature_xrange_.size();
                trained_ = true;
            }
        }
    }

    void SVMClassifier::save() const
    {
        if (trained_) {
            std::string save_name = "svm_" + std::to_string(svm_model_->nr_class) + "labels_";
            save_name += common::getCurrentTimestampString();

            const std::string& model_filename = params_.classifier_model_path + "/" + save_name + ".model";
            const std::string& range_filename = params_.classifier_model_path + "/" + save_name + ".range";

            // save SVM model
            std::fstream range_file;
            range_file.open(range_filename.c_str(), std::fstream::out);
            if (!range_file.is_open()) {
                ROS_WARN("Failed to save, can not save range file.");
            }
            else {
                if (svm_save_model(model_filename.c_str(), svm_model_)) {
                    ROS_WARN("Failed to save, can not save SVM model.");
                } else {
                    ROS_INFO("Saved SVM model into '%s'.", model_filename.c_str());
                    // 1 more size for end index (-1)
                    //svm_node_ = (struct svm_node *) malloc((FEATURE_SIZE + 1) * sizeof(struct svm_node));

                    // load range file, for more details: https://github.com/cjlin1/libsvm/
                    std::fstream range_file;
                    range_file.open(range_filename.c_str(), std::fstream::out);
                    if (!range_file.is_open()) {
                        ROS_WARN("Can not save range file.");
                    } else {
                        //save range name "x"
                        range_file << "x" << std::endl;
                        //save xrange lower/upper
                        range_file << params_.svm_feature_range_lower << " "
                                   << params_.svm_feature_range_upper << std::endl;
                        //get xrange for each feature
                        for (size_t f = 0u; f < svm_feature_size_; ++f) {
                            range_file << f+1 << " "
                                       << svm_feature_xrange_[f].min << " "
                                       << svm_feature_xrange_[f].max << std::endl;
                        }
                        ROS_INFO("Saved SVM range into '%s'.", range_filename.c_str());
                    }
                }
            }
        }
        else {
            ROS_WARN("An empty classifier, no need to save.");
        }
    }

    void SVMClassifier::feature2SvmNode(const Feature& feature,
                                        struct svm_feature_node* svm_feature_node)
    {
        if (svm_feature_node == nullptr) {
            return;
        }
        const unsigned int dim_feature = feature.size();
        const unsigned int dim_feature_node = (*svm_feature_node).size_-1;
        if (dim_feature_node != dim_feature) {
            ROS_ERROR_STREAM("Inconsistent in size, feature dimension is " << dim_feature
                                                                           << ", while svm feature node dimension is "
                                                                           << dim_feature_node << ".");
        }

        for (size_t f = 0u; f < dim_feature; ++f) {
            (*svm_feature_node).feature_node_[f].index = f + 1;
            (*svm_feature_node).feature_node_[f].value = feature.at(f).value;
        }

        // 1 more size for end index (-1)
        (*svm_feature_node).feature_node_[dim_feature_node].index = -1;
    }

    double SVMClassifier::normalizeFeatureValue(const double& value, const size_t& feature_index) const
    {
        double ret_val = 0.;
        if (svm_feature_xrange_[feature_index].min == svm_feature_xrange_[feature_index].max) {
            // skip single-valued attribute without normalizing
            //continue;
            ret_val = value;
        }
        if (value == svm_feature_xrange_[feature_index].min) {
            ret_val = params_.svm_feature_range_lower;
        }
        else if (value == svm_feature_xrange_[feature_index].max) {
            ret_val = params_.svm_feature_range_upper;
        }
        else {
            double xrange = svm_feature_xrange_[feature_index].max - svm_feature_xrange_[feature_index].min;
            ret_val = params_.svm_feature_range_lower + (params_.svm_feature_range_upper - params_.svm_feature_range_lower)
                                                      * (value - svm_feature_xrange_[feature_index].min) / xrange;
        }

        return ret_val;
    }

    /**
     * @result
     *  update svm parameters: C & gamma
     */
    void SVMClassifier::findBestTrainingParams()
    {
        std::ofstream s;
        s.open("svm_training_data");
        for (size_t i = 0u; i < svm_problem_.l; ++i) {
            s << svm_problem_.y[i];
            for (size_t j = 0u; j < svm_feature_size_; ++j)
                s << " " << svm_problem_.x[i][j].index << ":" << svm_problem_.x[i][j].value;
            s << "\n";
        }
        s.close();

        ROS_INFO("Finding the best training parameters ...");
        if (svm_check_parameter(&svm_problem_, &svm_parameter_) == NULL) {
            char result[100];
            FILE* fp = popen("./grid.py svm_training_data", "r");
            if (fp == NULL) {
                ROS_ERROR("Can not run cross validation!");
            } else {
                if (fgets(result, 100, fp) != NULL) {
                    char* pch = strtok(result, " ");
                    svm_parameter_.C = atof(pch);
                    pch = strtok(NULL, " ");
                    svm_parameter_.gamma = atof(pch);
                    pch = strtok(NULL, " ");
                    float rate = atof(pch);
                    ROS_INFO_STREAM("########## Best Training Parameters ##########\n"
                                            << "\tC: " << svm_parameter_.C << "\n"
                                            << "\tgamma: " << svm_parameter_.gamma << "\n"
                                            << "\tCV rate: " << rate);
                }
            }
            pclose(fp);
        }
    }

    svm_model* SVMClassifier::clone(const svm_model* rhs)
    {
        std::string config_file = params_.classifier_model_path + "/" + config_name_;
        svm_save_model(config_file.c_str(), rhs);
        return svm_load_model(config_file.c_str());
    }
}