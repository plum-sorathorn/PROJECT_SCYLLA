// ============================================================
// PROJECT: SCYLLA // C++ Core Engine
// main.cpp — Entry point. Starts Crow HTTP server on port 8080.
// Registers all routes and runs with multi-threaded concurrency.
// ============================================================
#include <crow.h>
#include "api_handlers.h"
#include <iostream>

int main() {
    crow::SimpleApp app;
    
    std::cout << R"(
  ____  ____   ___     _ _____ ____ _____      ____   ______   ____  _     _       _    
 |  _ \|  _ \ / _ \   | | ____/ ___|_   _|    / ___| / ___\ \ / /  | |   | |     / \   
 | |_) | |_) | | | |  | |  _|| |     | |      \___ \| |    \ V /|  | |   | |    / _ \  
 |  __/|  _ <| |_| |  | | |__| |___  | |       ___) | |___  | | |  | |___| |___/ ___ \ 
 |_|   |_| \_\\___/  _/ |_____\____| |_|      |____/ \____| |_| |  |_____|_____/_/   \_\
                     |__/                                                                 
  C++ CORE ENGINE v1.0 // CROW HTTP // PORT 8080
)" << std::endl;

    scylla::registerRoutes(app);

    std::cout << "[SCYLLA] C++ Core running on http://127.0.0.1:8080" << std::endl;
    std::cout << "[SCYLLA] Pulling data from Python ODP on http://127.0.0.1:6900" << std::endl;
    std::cout << "[SCYLLA] Frontend at http://127.0.0.1:8080/" << std::endl;

    app.port(8080)
       .multithreaded()
       .run();

    return 0;
}
