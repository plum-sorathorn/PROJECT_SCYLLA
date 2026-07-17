// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// api_handlers.cpp — Crow route implementations
// Serves processed data to the frontend on port 8080
// ============================================================
#include "api_handlers.h"
#include "data_fetcher.h"
#include "metrics_engine.h"
#include <crow.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace scylla {

// Helper: read file into string (for serving static HTML)
static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
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

    // CORS middleware helper
    auto addCors = [](crow::response& res) {
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Access-Control-Allow-Methods", "GET, OPTIONS");
        res.add_header("Access-Control-Allow-Headers", "Content-Type");
    };

    // ── Health check ──────────────────────────────────────────────────────────
    CROW_ROUTE(app, "/health")
    ([&addCors]() {
        crow::response res;
        addCors(res);
        json body = {{"status", "online"}, {"service", "SCYLLA C++ Core"}, {"port", 8080}};
        res.set_header("Content-Type", "application/json");
        res.body = body.dump();
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

    // ── Serve static frontend ──────────────────────────────────────────────────
    // The frontend build is placed in ../frontend/dist/
    CROW_ROUTE(app, "/")
    ([]() {
        std::string html = readFile("../frontend/dist/index.html");
        if (html.empty()) {
            return crow::response(404, "Frontend not built. Run: npm run build in /frontend");
        }
        crow::response res(html);
        res.set_header("Content-Type", "text/html; charset=utf-8");
        return res;
    });

    // Fallback route for SPA
    CROW_ROUTE(app, "/<path>")
    ([](const std::string& path) {
        std::string filePath = "../frontend/dist/" + path;
        std::string content = readFile(filePath);
        if (content.empty()) {
            content = readFile("../frontend/dist/index.html");
        }
        if (content.empty()) return crow::response(404, "Not found");
        crow::response res(content);
        // Set content type based on extension
        if (path.ends_with(".js"))  res.set_header("Content-Type", "application/javascript");
        else if (path.ends_with(".css")) res.set_header("Content-Type", "text/css");
        else if (path.ends_with(".html")) res.set_header("Content-Type", "text/html");
        else res.set_header("Content-Type", "application/octet-stream");
        return res;
    });
}

} // namespace scylla
