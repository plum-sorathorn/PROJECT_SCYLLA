#pragma once
// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// data_fetcher.h — HTTP client to pull JSON from Python ODP layer
// ============================================================
#include <string>
#include <vector>
#include <functional>

namespace scylla {

struct RawOptionRow {
    std::string ticker;
    std::string expiration;
    double strike;
    std::string optionType;
    long volume;
    long openInterest;
    double volOiRatio;
    double impliedVolatility;
    double underlierPrice;
    int    above50dSMA;   // 1=true, 0=false, -1=unknown
    int    above200dSMA;
    double expectedMove;
    int    dte;
    double premium;
    bool   isWeekly;
    std::string lastTradeDate;
    std::string side;
};

struct PcrPoint {
    std::string ticker;
    std::string date;
    double putCallRatio;
};

struct VolConcentrationPoint {
    std::string expiration;
    long callVolume;
    long putVolume;
};

struct IVSkewPoint {
    std::string expiration;
    double strike;
    double iv;
    std::string optionType;
};

struct IVSandboxResult {
    std::string ticker;
    double currentIV;
    double ivRank;
    double ivPercentile;
    std::vector<IVSkewPoint> smileData;
};

// Fetch functions — blocking HTTP GET to Python ODP on port 6900
std::vector<RawOptionRow>        fetchUnusualOptions(double minVolOI = 2.0);
std::vector<PcrPoint>            fetchPutCallRatio(const std::vector<std::string>& tickers);
std::vector<VolConcentrationPoint> fetchVolumeConcentration(const std::string& ticker = "SPY");
IVSandboxResult                  fetchIVSkew(const std::string& ticker);
std::string                      fetchTacticalBundle(double minVolOI, const std::string& volconTicker, const std::string& ivTicker);

} // namespace scylla
