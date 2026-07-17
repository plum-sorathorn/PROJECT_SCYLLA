#pragma once
// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// api_handlers.h — Crow route handler declarations
// ============================================================
#include <crow.h>

namespace scylla {
    void registerRoutes(crow::SimpleApp& app);
} // namespace scylla
