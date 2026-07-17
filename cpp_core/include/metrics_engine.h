#pragma once
// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// metrics_engine.h — Multi-threaded options metric computation
// ============================================================
#include "data_fetcher.h"
#include <vector>
#include <string>

namespace scylla {

struct ProcessedOptionRow : RawOptionRow {
    double normalizedVolOI;     // log-scaled for display weighting
    bool   isWhaleSignal;       // vol/OI > 5.0
    std::string trendAlignment; // "BULL_ALIGNED" | "BEAR_CONTRARIAN" | "NEUTRAL" | "UNKNOWN"
    double expectedMoveUpper;
    double expectedMoveLower;
};

struct MetricsSummary {
    double avgVolOI;
    double maxVolOI;
    long   totalCallVolume;
    long   totalPutVolume;
    double aggregatePCR;
    int    whaleSignalCount;
};

// Process raw rows in parallel (multi-threaded)
std::vector<ProcessedOptionRow> processOptionRows(std::vector<RawOptionRow> rawRows);

// Compute high-level summary metrics
MetricsSummary computeSummary(const std::vector<ProcessedOptionRow>& rows);

} // namespace scylla
