// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// metrics_engine.cpp — Multi-threaded options metric computation
// ============================================================
#include "metrics_engine.h"
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

// ------------------------------------------------------------
// SCYLLA PHASE B.1 — vendor link smoke test (LEAVE COMMENTED).
// Uncomment the block below on a machine with MSVC to verify
// that lib_lightgbm.dll and libcurl-x64.dll import correctly,
// then re-comment (or delete) it before committing B.3 logic.
// ------------------------------------------------------------
#if 0
#include <lightgbm/c_api.h>
#include <curl/curl.h>

[[maybe_unused]] static void scylla_phase_b1_smoke_test() {
    // LightGBM C API link check (no real model yet — Phase B.3).
    void* booster = nullptr;
    LGBM_BoosterCreate(0, 0, reinterpret_cast<BoosterHandle*>(&booster));
    LGBM_BoosterFree(static_cast<BoosterHandle>(booster));

    // libcurl link check (no real request yet — Phase B.3).
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_global_cleanup();
}
#endif

namespace scylla {

// Process a single row to produce enriched ProcessedOptionRow
static ProcessedOptionRow processRow(const RawOptionRow& raw) {
    ProcessedOptionRow p;
    // Copy base fields
    static_cast<RawOptionRow&>(p) = raw;

    // Normalize vol/OI on a log scale (cap at 9999 for inf cases)
    double ratio = (raw.volOiRatio > 9990.0) ? 50.0 : raw.volOiRatio;
    p.normalizedVolOI = std::log1p(ratio);

    // Whale signal: vol/OI > 5x
    p.isWhaleSignal = (raw.volOiRatio >= 5.0);

    // Expected move range
    p.expectedMoveUpper = raw.underlierPrice + raw.expectedMove;
    p.expectedMoveLower = raw.underlierPrice - raw.expectedMove;

    // Trend alignment: combine SMA flags with option type
    bool bullishTrend = (raw.above50dSMA == 1 && raw.above200dSMA != 0);
    bool bearishTrend = (raw.above50dSMA == 0 || raw.above200dSMA == 0);
    bool isCall = (raw.optionType == "Call");

    if (raw.above50dSMA == -1) {
        p.trendAlignment = "UNKNOWN";
    } else if (bullishTrend && isCall) {
        p.trendAlignment = "BULL_ALIGNED";
    } else if (bearishTrend && !isCall) {
        p.trendAlignment = "BEAR_ALIGNED";
    } else if (bullishTrend && !isCall) {
        p.trendAlignment = "BULL_CONTRARIAN";
    } else {
        p.trendAlignment = "NEUTRAL";
    }

    return p;
}

// Multi-threaded processing — splits work across hardware threads
std::vector<ProcessedOptionRow> processOptionRows(std::vector<RawOptionRow> rawRows) {
    size_t n = rawRows.size();
    std::vector<ProcessedOptionRow> result(n);

    unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
    size_t chunkSize = (n + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t start = t * chunkSize;
        size_t end   = std::min(start + chunkSize, n);
        if (start >= n) break;

        threads.emplace_back([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                result[i] = processRow(rawRows[i]);
            }
        });
    }
    for (auto& th : threads) th.join();

    // Sort by volOiRatio descending
    std::sort(result.begin(), result.end(), [](const ProcessedOptionRow& a, const ProcessedOptionRow& b) {
        return a.volOiRatio > b.volOiRatio;
    });

    return result;
}

MetricsSummary computeSummary(const std::vector<ProcessedOptionRow>& rows) {
    MetricsSummary s{};
    if (rows.empty()) return s;

    double sumRatio = 0.0;
    double maxRatio = 0.0;
    long callVol = 0, putVol = 0;
    int whales = 0;

    for (auto& r : rows) {
        double ratio = (r.volOiRatio > 9990.0) ? 9999.0 : r.volOiRatio;
        sumRatio += ratio;
        maxRatio = std::max(maxRatio, ratio);
        if (r.optionType == "Call") callVol += r.volume;
        else putVol += r.volume;
        if (r.isWhaleSignal) whales++;
    }

    s.avgVolOI          = sumRatio / rows.size();
    s.maxVolOI          = maxRatio;
    s.totalCallVolume   = callVol;
    s.totalPutVolume    = putVol;
    s.aggregatePCR      = (callVol > 0) ? (double)putVol / callVol : 1.0;
    s.whaleSignalCount  = whales;
    return s;
}

} // namespace scylla
