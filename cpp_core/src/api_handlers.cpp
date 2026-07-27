// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// api_handlers.cpp — Crow route implementations
// Serves processed data to the frontend on port 8080
// ============================================================
#include "api_handlers.h"
#include "data_fetcher.h"
#include "metrics_engine.h"
#include "inference_engine.h"
#include <crow.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <future>
#include <set>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace scylla {

static InferenceEngine g_engine;

// Helper: read file into string (for serving static HTML)
static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Helper: parse JSON object into PredictRowInput
static PredictRowInput parsePredictRowInput(const json& j) {
    PredictRowInput r;
    if (j.contains("ticker")) r.ticker = j["ticker"].get<std::string>();
    if (j.contains("underlier_price")) r.underlier_price = j["underlier_price"].get<double>();
    if (j.contains("strike")) r.strike = j["strike"].get<double>();
    if (j.contains("volume")) r.volume = j["volume"].get<double>();
    if (j.contains("open_interest")) r.open_interest = j["open_interest"].get<double>();
    if (j.contains("implied_vol")) r.implied_vol = j["implied_vol"].get<double>();
    if (j.contains("premium")) r.premium = j["premium"].get<double>();

    if (j.contains("side")) {
        if (j["side"].is_string()) r.side = j["side"].get<std::string>();
    }
    if (j.contains("dte")) r.dte = j["dte"].get<double>();

    if (j.contains("is_weekly")) {
        if (j["is_weekly"].is_string()) r.is_weekly = j["is_weekly"].get<std::string>();
        else if (j["is_weekly"].is_boolean()) r.is_weekly = j["is_weekly"].get<bool>() ? "True" : "False";
    }

    if (j.contains("trend_alignment")) {
        if (j["trend_alignment"].is_string()) r.trend_alignment = j["trend_alignment"].get<std::string>();
    }

    if (j.contains("synthetic_vol_ratio")) r.synthetic_vol_ratio = j["synthetic_vol_ratio"].get<double>();
    if (j.contains("synthetic_hist_vol")) r.synthetic_hist_vol = j["synthetic_hist_vol"].get<double>();
    if (j.contains("synthetic_vix")) r.synthetic_vix = j["synthetic_vix"].get<double>();

    return r;
}

// Helper: serialize processed row to JSON
static json rowToJson(const ProcessedOptionRow& r) {
    return json{
        {"ticker",            r.ticker},
        {"expiration",        r.expiration},
        {"strike",            r.strike},
        {"optionType",        r.optionType},
        {"volume",            r.volume},
        {"openInterest",      r.openInterest},
        {"volOiRatio",        r.volOiRatio},
        {"impliedVolatility", r.impliedVolatility},
        {"underlierPrice",    r.underlierPrice},
        {"above50dSMA",       r.above50dSMA},
        {"above200dSMA",      r.above200dSMA},
        {"expectedMove",      r.expectedMove},
        {"expectedMoveUpper", r.expectedMoveUpper},
        {"expectedMoveLower", r.expectedMoveLower},
        {"isWhaleSignal",     r.isWhaleSignal},
        {"trendAlignment",    r.trendAlignment},
        {"normalizedVolOI",   r.normalizedVolOI},
        {"dte",               r.dte},
        {"premium",           r.premium},
        {"isWeekly",          r.isWeekly},
        {"lastTradeDate",     r.lastTradeDate},
        {"side",              r.side},
    };
}

void registerRoutes(crow::SimpleApp& app) {

    // Load C++ Inference Engine artifacts from backend cache
    if (!g_engine.is_loaded()) {
        std::vector<std::string> candidate_paths = {
            "backend/cache/cpp_inference",
            "../backend/cache/cpp_inference",
            "../../backend/cache/cpp_inference"
        };
        for (const auto& path : candidate_paths) {
            if (g_engine.load(path)) {
                std::cout << "[SCYLLA] InferenceEngine successfully loaded artifacts from: " << path << std::endl;
                break;
            }
        }
        if (!g_engine.is_loaded()) {
            std::cerr << "[SCYLLA] Warning: InferenceEngine load failed across all candidate paths." << std::endl;
        }
    }

    // CORS middleware helper
    auto addCors = [](crow::response& res) {
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.add_header("Access-Control-Allow-Headers", "Content-Type");
    };

    // ── Health check ──────────────────────────────────────────────────────────
    CROW_ROUTE(app, "/health")
    ([&addCors]() {
        crow::response res;
        addCors(res);
        json body = {
            {"status", "online"},
            {"service", "SCYLLA C++ Core"},
            {"port", 8080},
            {"inference_engine_loaded", g_engine.is_loaded()}
        };
        res.set_header("Content-Type", "application/json");
        res.body = body.dump();
        return res;
    });

    // ── C++ Native ML Predict (Single-row) ───────────────────────────────────
    CROW_ROUTE(app, "/api/v1/ml/predict")
    .methods(crow::HTTPMethod::POST, crow::HTTPMethod::OPTIONS)
    ([&addCors](const crow::request& req) {
        crow::response res;
        addCors(res);
        if (req.method == crow::HTTPMethod::OPTIONS) return res;

        if (!g_engine.is_loaded()) {
            res.code = 503;
            res.body = json{{"error", "InferenceEngine not loaded"}}.dump();
            return res;
        }

        try {
            json body = json::parse(req.body);
            PredictRowInput input = parsePredictRowInput(body);

            double hv = (input.synthetic_hist_vol >= 0.0) ? input.synthetic_hist_vol : g_engine.fetch_hv(input.ticker);
            double vix = (input.synthetic_vix >= 0.0) ? input.synthetic_vix : g_engine.fetch_vix();

            auto vec = g_engine.vectorize_features(input, hv, vix);
            auto q = g_engine.predict_quantiles(vec);
            auto strat = g_engine.derive_strategy(q, input);

            json resp_json = {
                {"quantiles", q.quantiles},
                {"p_success", strat.p_success},
                {"expected_return", strat.expected_return},
                {"strategy", strat.strategy},
                {"kelly_fraction", strat.kelly_fraction},
                {"rejection_reason", strat.rejection_reason},
                {"direction_confidence", q.direction_confidence},
                {"iqr", q.iqr}
            };

            res.set_header("Content-Type", "application/json");
            res.body = resp_json.dump();
        } catch (const std::exception& e) {
            res.code = 400;
            res.body = json{{"error", e.what()}}.dump();
        }
        return res;
    });

    // ── C++ Native ML Predict (Batch) ─────────────────────────────────────────
    CROW_ROUTE(app, "/api/v1/ml/predict-batch")
    .methods(crow::HTTPMethod::POST, crow::HTTPMethod::OPTIONS)
    ([&addCors](const crow::request& req) {
        crow::response res;
        addCors(res);
        if (req.method == crow::HTTPMethod::OPTIONS) return res;

        if (!g_engine.is_loaded()) {
            res.code = 503;
            res.body = json{{"error", "InferenceEngine not loaded"}}.dump();
            return res;
        }

        try {
            json body = json::parse(req.body);
            if (!body.contains("rows") || !body["rows"].is_array()) {
                res.code = 400;
                res.body = json{{"error", "Missing 'rows' array in JSON body"}}.dump();
                return res;
            }

            const auto& rows_json = body["rows"];
            size_t n = rows_json.size();

            std::vector<PredictRowInput> inputs(n);
            std::set<std::string> unique_tickers;

            for (size_t i = 0; i < n; ++i) {
                inputs[i] = parsePredictRowInput(rows_json[i]);
                if (inputs[i].synthetic_hist_vol < 0.0 && !inputs[i].ticker.empty()) {
                    unique_tickers.insert(inputs[i].ticker);
                }
            }

            // Parallel HV fetch across unique tickers
            std::unordered_map<std::string, double> hv_map;
            std::vector<std::future<std::pair<std::string, double>>> hv_futures;
            hv_futures.reserve(unique_tickers.size());

            for (const auto& ticker : unique_tickers) {
                hv_futures.push_back(std::async(std::launch::async, [ticker]() {
                    return std::make_pair(ticker, g_engine.fetch_hv(ticker));
                }));
            }

            for (auto& fut : hv_futures) {
                auto p = fut.get();
                hv_map[p.first] = p.second;
            }

            double vix = g_engine.fetch_vix();

            // Vectorize feature matrix
            std::vector<std::vector<double>> matrix(n);
            for (size_t i = 0; i < n; ++i) {
                double hv = (inputs[i].synthetic_hist_vol >= 0.0) ? inputs[i].synthetic_hist_vol : hv_map[inputs[i].ticker];
                double row_vix = (inputs[i].synthetic_vix >= 0.0) ? inputs[i].synthetic_vix : vix;
                matrix[i] = g_engine.vectorize_features(inputs[i], hv, row_vix);
            }

            // Batch prediction matrix run
            auto batch_quantiles = g_engine.predict_quantiles_batch(matrix);

            json predictions_arr = json::array();
            for (size_t i = 0; i < n; ++i) {
                const auto& q = batch_quantiles[i];
                auto strat = g_engine.derive_strategy(q, inputs[i]);
                predictions_arr.push_back({
                    {"quantiles", q.quantiles},
                    {"p_success", strat.p_success},
                    {"expected_return", strat.expected_return},
                    {"strategy", strat.strategy},
                    {"kelly_fraction", strat.kelly_fraction},
                    {"rejection_reason", strat.rejection_reason},
                    {"direction_confidence", q.direction_confidence},
                    {"iqr", q.iqr}
                });
            }

            res.set_header("Content-Type", "application/json");
            res.body = json{{"predictions", predictions_arr}}.dump();
        } catch (const std::exception& e) {
            res.code = 400;
            res.body = json{{"error", e.what()}}.dump();
        }
        return res;
    });

    // ── Unusual options scanner (processed + enriched) ─────────────────────────
    CROW_ROUTE(app, "/api/scanner")
    .methods(crow::HTTPMethod::GET)
    ([&addCors](const crow::request& req) {
        crow::response res;
        addCors(res);
        try {
            double minVolOI = 2.0;
            auto param = req.url_params.get("min_vol_oi");
            if (param) {
                try {
                    minVolOI = std::stod(param);
                } catch (...) {}
            }
            auto raw = fetchUnusualOptions(minVolOI);
            auto processed = processOptionRows(raw);
            auto summary = computeSummary(processed);

            json rows = json::array();
            for (auto& r : processed) rows.push_back(rowToJson(r));

            json body = {
                {"data", rows},
                {"summary", {
                    {"avgVolOI",         summary.avgVolOI},
                    {"maxVolOI",         summary.maxVolOI},
                    {"totalCallVolume",  summary.totalCallVolume},
                    {"totalPutVolume",   summary.totalPutVolume},
                    {"aggregatePCR",     summary.aggregatePCR},
                    {"whaleSignalCount", summary.whaleSignalCount},
                }},
                {"count", rows.size()}
            };
            res.set_header("Content-Type", "application/json");
            res.body = body.dump();
        } catch (std::exception& e) {
            res.code = 502;
            res.body = json{{"error", e.what()}}.dump();
        }
        return res;
    });

    // ── Put/Call ratio ─────────────────────────────────────────────────────────
    CROW_ROUTE(app, "/api/put-call-ratio")
    .methods(crow::HTTPMethod::GET)
    ([&addCors](const crow::request& req) {
        crow::response res;
        addCors(res);
        try {
            auto pts = fetchPutCallRatio({"SPY", "QQQ", "IWM"});
            json out;
            out["SPY"] = json::array();
            out["QQQ"] = json::array();
            out["IWM"] = json::array();
            for (auto& p : pts) {
                out[p.ticker].push_back({{"date", p.date}, {"putCallRatio", p.putCallRatio}});
            }
            res.set_header("Content-Type", "application/json");
            res.body = json{{"data", out}}.dump();
        } catch (std::exception& e) {
            res.code = 502;
            res.body = json{{"error", e.what()}}.dump();
        }
        return res;
    });

    // ── Volume concentration by expiration ─────────────────────────────────────
    CROW_ROUTE(app, "/api/volume-concentration")
    .methods(crow::HTTPMethod::GET)
    ([&addCors](const crow::request& req) {
        crow::response res;
        addCors(res);
        try {
            std::string ticker = "SPY";
            auto param = req.url_params.get("ticker");
            if (param) ticker = std::string(param);
            auto pts = fetchVolumeConcentration(ticker);
            json arr = json::array();
            for (auto& p : pts) {
                arr.push_back({
                    {"expiration", p.expiration},
                    {"callVolume", p.callVolume},
                    {"putVolume",  p.putVolume},
                });
            }
            res.set_header("Content-Type", "application/json");
            res.body = json{{"ticker", ticker}, {"data", arr}}.dump();
        } catch (std::exception& e) {
            res.code = 502;
            res.body = json{{"error", e.what()}}.dump();
        }
        return res;
    });

    // ── IV Skew / Sandbox ──────────────────────────────────────────────────────
    CROW_ROUTE(app, "/api/iv-skew")
    .methods(crow::HTTPMethod::GET)
    ([&addCors](const crow::request& req) {
        crow::response res;
        addCors(res);
        try {
            std::string ticker = "SPY";
            auto param = req.url_params.get("ticker");
            if (param) ticker = std::string(param);
            auto result = fetchIVSkew(ticker);
            json smile = json::array();
            for (auto& sp : result.smileData) {
                smile.push_back({
                    {"expiration", sp.expiration},
                    {"strike",     sp.strike},
                    {"iv",         sp.iv},
                    {"optionType", sp.optionType},
                });
            }
            json body = {
                {"ticker",       result.ticker},
                {"currentIV",    result.currentIV},
                {"ivRank",       result.ivRank},
                {"ivPercentile", result.ivPercentile},
                {"smileData",    smile},
            };
            res.set_header("Content-Type", "application/json");
            res.body = body.dump();
        } catch (std::exception& e) {
            res.code = 502;
            res.body = json{{"error", e.what()}}.dump();
        }
        return res;
    });

    // Helper: resolve relative path to frontend asset across standard locations
    auto resolveFrontendPath = [](const std::string& relativeFile) -> std::string {
        std::vector<std::string> candidates = {
            "../frontend/" + relativeFile,
            "frontend/" + relativeFile,
            "../../frontend/" + relativeFile,
            "../frontend/dist/" + relativeFile,
            "frontend/dist/" + relativeFile
        };
        for (const auto& path : candidates) {
            if (fs::exists(path)) {
                return path;
            }
        }
        return "";
    };

    // ── Serve static frontend ──────────────────────────────────────────────────
    CROW_ROUTE(app, "/")
    ([resolveFrontendPath]() {
        std::string path = resolveFrontendPath("index.html");
        std::string html = readFile(path);
        if (html.empty()) {
            return crow::response(404, "Frontend asset index.html not found in frontend/ directory.");
        }
        crow::response res(html);
        res.set_header("Content-Type", "text/html; charset=utf-8");
        return res;
    });

    // Fallback route for static assets & SPA
    CROW_ROUTE(app, "/<path>")
    ([resolveFrontendPath](const std::string& subpath) {
        std::string filePath = resolveFrontendPath(subpath);
        std::string content = readFile(filePath);
        if (content.empty()) {
            std::string indexPath = resolveFrontendPath("index.html");
            content = readFile(indexPath);
        }
        if (content.empty()) return crow::response(404, "Asset not found");
        crow::response res(content);
        auto endsWith = [](const std::string& str, const std::string& suffix) {
            return str.size() >= suffix.size() &&
                   str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        if (endsWith(subpath, ".js"))  res.set_header("Content-Type", "application/javascript");
        else if (endsWith(subpath, ".css")) res.set_header("Content-Type", "text/css");
        else if (endsWith(subpath, ".html")) res.set_header("Content-Type", "text/html");
        else if (endsWith(subpath, ".png")) res.set_header("Content-Type", "image/png");
        else if (endsWith(subpath, ".svg")) res.set_header("Content-Type", "image/svg+xml");
        else res.set_header("Content-Type", "application/octet-stream");
        return res;
    });
}

} // namespace scylla
