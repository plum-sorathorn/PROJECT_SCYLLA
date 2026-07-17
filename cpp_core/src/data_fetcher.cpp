// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// data_fetcher.cpp — HTTP client impl using WinHTTP (Windows native)
// Pulls JSON arrays from Python ODP on http://127.0.0.1:6900
// ============================================================
#include "data_fetcher.h"
#include <nlohmann/json.hpp>
#include <windows.h>
#include <winhttp.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#pragma comment(lib, "winhttp.lib")

using json = nlohmann::json;

namespace scylla {

// --- Low-level WinHTTP GET helper ---
static std::string httpGet(const std::wstring& host, const std::wstring& path, INTERNET_PORT port = 6900) {
    HINTERNET hSession = WinHttpOpen(L"ScyllaCore/1.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) throw std::runtime_error("WinHttpOpen failed");

    HINTERNET hConnect = WinHttpConnect(hSession, host.c_str(), port, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); throw std::runtime_error("WinHttpConnect failed"); }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", path.c_str(),
        NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    if (!hRequest) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        throw std::runtime_error("WinHttpOpenRequest failed");
    }

    BOOL bResult = WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
        WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (!bResult || !WinHttpReceiveResponse(hRequest, NULL)) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        throw std::runtime_error("HTTP request failed");
    }

    std::string response;
    DWORD dwSize = 0;
    do {
        dwSize = 0;
        if (!WinHttpQueryDataAvailable(hRequest, &dwSize)) break;
        if (dwSize == 0) break;
        std::vector<char> buffer(dwSize + 1, 0);
        DWORD dwDownloaded = 0;
        WinHttpReadData(hRequest, buffer.data(), dwSize, &dwDownloaded);
        response.append(buffer.data(), dwDownloaded);
    } while (dwSize > 0);

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return response;
}

// Helper: safely get int from json (handles bool/null)
static int safeInt(const json& j) {
    if (j.is_null()) return -1;
    if (j.is_boolean()) return j.get<bool>() ? 1 : 0;
    if (j.is_number()) return j.get<int>();
    return -1;
}

static double safeDouble(const json& j, double def = 0.0) {
    if (j.is_null()) return def;
    if (j.is_number()) return j.get<double>();
    return def;
}

// ---- Fetch unusual options ----
std::vector<RawOptionRow> fetchUnusualOptions(double minVolOI) {
    std::wstring path = L"/api/v1/unusual-options?min_vol_oi=" + std::to_wstring(minVolOI) + L"&limit=200";
    std::string body = httpGet(L"127.0.0.1", path);
    auto parsed = json::parse(body);
    std::vector<RawOptionRow> rows;
    for (auto& item : parsed["data"]) {
        RawOptionRow r;
        r.ticker            = item.value("ticker", "");
        r.expiration        = item.value("expiration", "");
        r.strike            = safeDouble(item["strike"]);
        r.optionType        = item.value("optionType", "");
        r.volume            = item.value("volume", 0);
        r.openInterest      = item.value("openInterest", 0);
        r.volOiRatio        = safeDouble(item["volOiRatio"]);
        r.impliedVolatility = safeDouble(item["impliedVolatility"]);
        r.underlierPrice    = safeDouble(item["underlierPrice"]);
        r.above50dSMA       = safeInt(item["above50dSMA"]);
        r.above200dSMA      = safeInt(item["above200dSMA"]);
        r.expectedMove      = safeDouble(item["expectedMove"], 0.0);
        rows.push_back(r);
    }
    return rows;
}

// ---- Fetch put/call ratio ----
std::vector<PcrPoint> fetchPutCallRatio(const std::vector<std::string>& tickers) {
    std::wstring tickerParam;
    for (size_t i = 0; i < tickers.size(); i++) {
        if (i > 0) tickerParam += L",";
        tickerParam += std::wstring(tickers[i].begin(), tickers[i].end());
    }
    std::wstring path = L"/api/v1/put-call-ratio?tickers=" + tickerParam;
    std::string body = httpGet(L"127.0.0.1", path);
    auto parsed = json::parse(body);
    std::vector<PcrPoint> points;
    for (auto& [ticker, arr] : parsed["data"].items()) {
        for (auto& item : arr) {
            PcrPoint p;
            p.ticker = ticker;
            p.date = item.value("date", "");
            p.putCallRatio = safeDouble(item["putCallRatio"]);
            points.push_back(p);
        }
    }
    return points;
}

// ---- Fetch volume concentration ----
std::vector<VolConcentrationPoint> fetchVolumeConcentration(const std::string& ticker) {
    std::wstring t(ticker.begin(), ticker.end());
    std::wstring path = L"/api/v1/volume-concentration?ticker=" + t;
    std::string body = httpGet(L"127.0.0.1", path);
    auto parsed = json::parse(body);
    std::vector<VolConcentrationPoint> pts;
    for (auto& item : parsed["data"]) {
        VolConcentrationPoint p;
        p.expiration  = item.value("expiration", "");
        p.callVolume  = item.value("callVolume", 0LL);
        p.putVolume   = item.value("putVolume", 0LL);
        pts.push_back(p);
    }
    return pts;
}

// ---- Fetch IV skew ----
IVSandboxResult fetchIVSkew(const std::string& ticker) {
    std::wstring t(ticker.begin(), ticker.end());
    std::wstring path = L"/api/v1/iv-skew?ticker=" + t;
    std::string body = httpGet(L"127.0.0.1", path);
    auto parsed = json::parse(body);
    IVSandboxResult res;
    res.ticker        = parsed.value("ticker", "");
    res.currentIV     = safeDouble(parsed["currentIV"]);
    res.ivRank        = safeDouble(parsed["ivRank"]);
    res.ivPercentile  = safeDouble(parsed["ivPercentile"]);
    for (auto& pt : parsed["smileData"]) {
        IVSkewPoint sp;
        sp.expiration = pt.value("expiration", "");
        sp.strike     = safeDouble(pt["strike"]);
        sp.iv         = safeDouble(pt["iv"]);
        sp.optionType = pt.value("optionType", "");
        res.smileData.push_back(sp);
    }
    return res;
}

} // namespace scylla
