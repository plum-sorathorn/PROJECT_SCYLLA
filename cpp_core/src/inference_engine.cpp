#include "inference_engine.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <future>
#include <sstream>
#include <chrono>

#include <curl/curl.h>

namespace scylla {

namespace {

// Helper libcurl write callback
size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), total_size);
    return total_size;
}

std::string HttpGet(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";

    std::string read_buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L); // 5s timeout

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) return "";
    return read_buffer;
}

} // namespace

InferenceEngine::InferenceEngine() = default;

InferenceEngine::~InferenceEngine() {
    for (int i = 0; i < 5; ++i) {
        if (boosters_[i]) {
            LGBM_BoosterFree(boosters_[i]);
            boosters_[i] = nullptr;
        }
    }
}

bool InferenceEngine::load(const std::string& artifact_dir) {
    std::string json_path = artifact_dir + "/scylla_preprocessor.json";
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "[InferenceEngine] Could not open preprocessor schema: " << json_path << std::endl;
        return false;
    }

    try {
        nlohmann::json meta = nlohmann::json::parse(f);
        numeric_features_ = meta["numeric_features"].get<std::vector<std::string>>();
        categorical_features_ = meta["categorical_features"].get<std::vector<std::string>>();
        numeric_medians_ = meta["numeric_medians"].get<std::vector<double>>();
        ohe_categories_ = meta["ohe_categories"].get<std::vector<std::vector<std::string>>>();

        total_feature_dim_ = static_cast<int>(numeric_features_.size());
        for (const auto& cat_list : ohe_categories_) {
            total_feature_dim_ += static_cast<int>(cat_list.size());
        }

        const int q_ints[5] = {10, 25, 50, 75, 90};
        for (int i = 0; i < 5; ++i) {
            std::string model_path = artifact_dir + "/scylla_q" + std::to_string(q_ints[i]) + ".txt";
            int out_num_iterations = 0;
            int err = LGBM_BoosterCreateFromModelfile(model_path.c_str(), &out_num_iterations, &boosters_[i]);
            if (err != 0 || !boosters_[i]) {
                std::cerr << "[InferenceEngine] Failed to load LightGBM Booster: " << model_path << std::endl;
                return false;
            }
        }

        is_loaded_ = true;
        std::cout << "[InferenceEngine] Loaded 5 quantile boosters. Feature dimension = " << total_feature_dim_ << std::endl;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[InferenceEngine] Schema parse error: " << ex.what() << std::endl;
        return false;
    }
}

std::vector<double> InferenceEngine::vectorize_features(
    const PredictRowInput& input,
    double resolved_hv,
    double resolved_vix
) const {
    std::unordered_map<std::string, double> raw_num;
    raw_num["underlier_price"] = input.underlier_price;
    raw_num["strike"] = input.strike;
    raw_num["volume"] = input.volume;
    raw_num["open_interest"] = input.open_interest;
    raw_num["implied_vol"] = input.implied_vol;
    raw_num["premium"] = input.premium;
    raw_num["dte"] = input.dte;

    double vol_oi = input.volume / (input.open_interest + 1e-6);
    raw_num["vol_oi_ratio"] = vol_oi;

    double moneyness = (input.underlier_price > 0.0) ? (input.strike / input.underlier_price) : 1.0;
    raw_num["moneyness"] = moneyness;

    double iv_hv = input.implied_vol / (resolved_hv + 1e-6);
    raw_num["iv_hv_ratio"] = iv_hv;

    raw_num["hist_vol"] = resolved_hv;
    raw_num["vix_level"] = resolved_vix;

    std::vector<double> vec;
    vec.reserve(total_feature_dim_);

    // 1. Numeric features with median imputation
    for (size_t i = 0; i < numeric_features_.size(); ++i) {
        const auto& name = numeric_features_[i];
        auto it = raw_num.find(name);
        double val = (it != raw_num.end()) ? it->second : numeric_medians_[i];
        if (std::isnan(val) || std::isinf(val)) {
            val = numeric_medians_[i];
        }
        vec.push_back(val);
    }

    // 2. Categorical features with One-Hot Encoding
    std::unordered_map<std::string, std::string> raw_cat;
    raw_cat["ticker"] = input.ticker;
    raw_cat["side"] = input.side;
    raw_cat["is_weekly"] = input.is_weekly;
    raw_cat["trend_alignment"] = input.trend_alignment;

    for (size_t i = 0; i < categorical_features_.size(); ++i) {
        const auto& cat_name = categorical_features_[i];
        const auto& valid_cats = ohe_categories_[i];
        std::string val = raw_cat.count(cat_name) ? raw_cat.at(cat_name) : "";

        for (const auto& category_label : valid_cats) {
            vec.push_back((val == category_label) ? 1.0 : 0.0);
        }
    }

    return vec;
}

QuantilePrediction InferenceEngine::predict_quantiles(const std::vector<double>& feature_vec) const {
    return predict_single_booster_set(feature_vec);
}

QuantilePrediction InferenceEngine::predict_single_booster_set(const std::vector<double>& feature_vec) const {
    QuantilePrediction pred;
    if (!is_loaded_ || feature_vec.size() != static_cast<size_t>(total_feature_dim_)) {
        return pred;
    }

    std::array<double, 5> raw_preds;
    for (int i = 0; i < 5; ++i) {
        int64_t out_len = 0;
        double out_result = 0.0;
        int err = LGBM_BoosterPredictForMat(
            boosters_[i],
            feature_vec.data(),
            C_API_DTYPE_FLOAT64,
            1,
            total_feature_dim_,
            1, // col_major = row_major (1 row)
            C_API_PREDICT_NORMAL,
            0,
            -1,
            "",
            &out_len,
            &out_result
        );
        raw_preds[i] = (err == 0) ? out_result : 0.0;
    }

    // Monotonicity sorting
    std::sort(raw_preds.begin(), raw_preds.end());

    pred.p10 = raw_preds[0];
    pred.p25 = raw_preds[1];
    pred.p50 = raw_preds[2];
    pred.p75 = raw_preds[3];
    pred.p90 = raw_preds[4];

    pred.quantiles["p10"] = pred.p10;
    pred.quantiles["p25"] = pred.p25;
    pred.quantiles["p50"] = pred.p50;
    pred.quantiles["p75"] = pred.p75;
    pred.quantiles["p90"] = pred.p90;

    pred.iqr = pred.p75 - pred.p25;
    pred.direction_confidence = (pred.p50 - pred.p10) / (pred.iqr + 1e-6);

    return pred;
}

std::vector<QuantilePrediction> InferenceEngine::predict_quantiles_batch(
    const std::vector<std::vector<double>>& feature_matrix
) const {
    size_t num_rows = feature_matrix.size();
    std::vector<QuantilePrediction> results(num_rows);

    if (num_rows == 0) return results;

    // Parallel quantile predictions across matrix rows using std::async
    std::vector<std::future<void>> futures;
    futures.reserve(num_rows);

    for (size_t r = 0; r < num_rows; ++r) {
        futures.push_back(std::async(std::launch::async, [this, &feature_matrix, &results, r]() {
            results[r] = predict_single_booster_set(feature_matrix[r]);
        }));
    }

    for (auto& fut : futures) {
        fut.get();
    }

    return results;
}

StrategyOutput InferenceEngine::derive_strategy(
    const QuantilePrediction& q,
    const PredictRowInput& input,
    double profit_threshold,
    double calibration_target,
    double iqr_threshold,
    double direction_threshold
) const {
    StrategyOutput out;
    out.raw_quantiles = q;
    out.expected_return = q.p50;

    // 1. Compute p_success using Gaussian CDF approximation over quantile range
    double mu = q.p50;
    double sigma = (q.p75 - q.p25) / 1.349;
    if (sigma <= 1e-6) sigma = 1e-6;

    double z = (profit_threshold - mu) / sigma;
    double p_succ = 1.0 - 0.5 * std::erfc(-z / std::sqrt(2.0));

    // Calibration shift & bounds clipping
    p_succ += calibration_target;
    if (p_succ < 0.02) p_succ = 0.02;
    if (p_succ > 0.98) p_succ = 0.98;
    out.p_success = p_succ;

    // 2. Sizing: Kelly Criterion with 0.05 cap
    double kelly_cap = 0.05;
    double b = (out.expected_return > 0.0) ? out.expected_return : 0.08;
    double kelly = (out.p_success * (1.0 + b) - 1.0) / (b + 1e-6);
    if (kelly < 0.0) kelly = 0.0;
    if (kelly > kelly_cap) kelly = kelly_cap;
    out.kelly_fraction = kelly;

    // 3. Strategy classification filters
    double vol_oi = input.volume / (input.open_interest + 1e-6);
    bool valid_vol_oi = (vol_oi >= 3.0 && vol_oi <= 50.0);
    bool valid_iv = (input.implied_vol >= 15.0 && input.implied_vol <= 150.0);
    bool valid_dte = (input.dte >= 14.0 && input.dte <= 30.0);
    bool quality_signal = (out.p_success >= direction_threshold && q.iqr <= iqr_threshold && q.p50 >= 0.04);

    if (valid_vol_oi && valid_iv && valid_dte && quality_signal) {
        out.strategy = "whale_quality";
    } else if (valid_vol_oi && valid_iv && valid_dte &&
               ((input.trend_alignment == "BULL_ALIGNED" && input.side == "PUT") ||
                (input.trend_alignment == "BEAR_ALIGNED" && input.side == "CALL")) &&
               quality_signal) {
        out.strategy = "contrarian_trend";
    } else if (valid_vol_oi && valid_dte &&
               ((input.implied_vol < 25.0 && out.p_success >= 0.60) ||
                (input.implied_vol > 50.0 && q.iqr <= 0.30))) {
        out.strategy = "vol_regime";
    } else {
        out.strategy = "NONE";
        out.rejection_reason = "Filters did not meet quality thresholds";
    }

    return out;
}

double InferenceEngine::fetch_hv(const std::string& ticker) const {
    std::lock_guard<std::mutex> lock(hv_cache_mutex_);
    auto it = hv_cache_.find(ticker);
    if (it != hv_cache_.end()) {
        return it->second;
    }

    std::string url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker + "?range=3mo&interval=1d";
    std::string json_body = HttpGet(url);

    double calculated_hv = 25.0; // Fallback HV
    if (!json_body.empty()) {
        try {
            auto res = nlohmann::json::parse(json_body);
            const auto& closes = res["chart"]["result"][0]["indicators"]["quote"][0]["close"];
            std::vector<double> valid_closes;
            for (const auto& val : closes) {
                if (!val.is_null()) {
                    valid_closes.push_back(val.get<double>());
                }
            }

            if (valid_closes.size() >= 10) {
                std::vector<double> log_returns;
                for (size_t i = 1; i < valid_closes.size(); ++i) {
                    if (valid_closes[i - 1] > 0.0 && valid_closes[i] > 0.0) {
                        log_returns.push_back(std::log(valid_closes[i] / valid_closes[i - 1]));
                    }
                }

                if (!log_returns.empty()) {
                    double mean = 0.0;
                    for (double r : log_returns) mean += r;
                    mean /= log_returns.size();

                    double sq_sum = 0.0;
                    for (double r : log_returns) sq_sum += (r - mean) * (r - mean);
                    double std_dev = std::sqrt(sq_sum / log_returns.size());

                    calculated_hv = std_dev * std::sqrt(252.0) * 100.0;
                }
            }
        } catch (...) {
            // Keep fallback
        }
    }

    hv_cache_[ticker] = calculated_hv;
    return calculated_hv;
}

double InferenceEngine::fetch_vix() const {
    std::lock_guard<std::mutex> lock(vix_cache_mutex_);
    uint64_t now_ts = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    if (vix_last_fetched_ts_ > 0 && (now_ts - vix_last_fetched_ts_) < 300) {
        return cached_vix_;
    }

    std::string url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?range=5d&interval=1d";
    std::string json_body = HttpGet(url);

    if (!json_body.empty()) {
        try {
            auto res = nlohmann::json::parse(json_body);
            const auto& closes = res["chart"]["result"][0]["indicators"]["quote"][0]["close"];
            for (auto it = closes.rbegin(); it != closes.rend(); ++it) {
                if (!it->is_null()) {
                    cached_vix_ = it->get<double>();
                    vix_last_fetched_ts_ = now_ts;
                    break;
                }
            }
        } catch (...) {}
    }

    return cached_vix_;
}

} // namespace scylla
