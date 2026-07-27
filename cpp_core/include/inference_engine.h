#ifndef SCYLLA_INFERENCE_ENGINE_H
#define SCYLLA_INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <array>

#include <LightGBM/c_api.h>
#include <nlohmann/json.hpp>

namespace scylla {

struct QuantilePrediction {
    std::unordered_map<std::string, double> quantiles; // "p10", "p25", "p50", "p75", "p90"
    double p10 = 0.0;
    double p25 = 0.0;
    double p50 = 0.0;
    double p75 = 0.0;
    double p90 = 0.0;
    double iqr = 0.0;
    double direction_confidence = 0.0;
};

struct PredictRowInput {
    std::string ticker;
    double underlier_price = 0.0;
    double strike = 0.0;
    double volume = 0.0;
    double open_interest = 0.0;
    double implied_vol = 0.0;
    double premium = 0.0;
    std::string side = "CALL"; // "CALL" or "PUT"
    double dte = 0.0;
    std::string is_weekly = "False"; // "True" or "False"
    std::string trend_alignment = "NEUTRAL"; // "BULL_ALIGNED", "BEAR_ALIGNED", "NEUTRAL"
    double synthetic_vol_ratio = -1.0; // If >= 0, overrides live fetch
    double synthetic_hist_vol = -1.0;
    double synthetic_vix = -1.0;
};

struct StrategyOutput {
    double p_success = 0.0;
    double expected_return = 0.0;
    std::string strategy = "NONE";
    double kelly_fraction = 0.0;
    std::string rejection_reason;
    QuantilePrediction raw_quantiles;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Load raw LightGBM Boosters (scylla_q*.txt) and preprocessor mapping (scylla_preprocessor.json)
    bool load(const std::string& artifact_dir);

    bool is_loaded() const { return is_loaded_; }

    // Vectorize single input row into continuous feature vector
    std::vector<double> vectorize_features(
        const PredictRowInput& input,
        double resolved_hv,
        double resolved_vix
    ) const;

    // Single-row quantile inference
    QuantilePrediction predict_quantiles(const std::vector<double>& feature_vec) const;

    // Batch quantile inference (parallel across quantiles & rows)
    std::vector<QuantilePrediction> predict_quantiles_batch(
        const std::vector<std::vector<double>>& feature_matrix
    ) const;

    // Ported strategy & risk derivation math from ml_derivations.py
    StrategyOutput derive_strategy(
        const QuantilePrediction& q,
        const PredictRowInput& input,
        double profit_threshold = 0.50,
        double calibration_target = 0.025,
        double iqr_threshold = 0.20,
        double direction_threshold = 0.55
    ) const;

    // Historical volatility fetch (cached in memory)
    double fetch_hv(const std::string& ticker) const;

    // VIX index fetch (5-minute TTL cache)
    double fetch_vix() const;

private:
    bool is_loaded_ = false;
    BoosterHandle boosters_[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    std::vector<double> quantiles_keys_ = {0.10, 0.25, 0.50, 0.75, 0.90};

    // Preprocessor schema
    std::vector<std::string> numeric_features_;
    std::vector<std::string> categorical_features_;
    std::vector<double> numeric_medians_;
    std::vector<std::vector<std::string>> ohe_categories_;
    int total_feature_dim_ = 0;

    // Caching
    mutable std::mutex hv_cache_mutex_;
    mutable std::unordered_map<std::string, double> hv_cache_;

    mutable std::mutex vix_cache_mutex_;
    mutable double cached_vix_ = 20.0;
    mutable uint64_t vix_last_fetched_ts_ = 0;

    QuantilePrediction predict_single_booster_set(const std::vector<double>& feature_vec) const;
};

} // namespace scylla

#endif // SCYLLA_INFERENCE_ENGINE_H
