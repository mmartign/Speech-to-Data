// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Data project.
//
// Copyright (C) 2025 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see https://www.gnu.org/licenses/.
//
// SPDX-License-Identifier: GPL-2.0-or-later
// Spazio IT Speech-to-Data C++ version
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <openai.hpp>

using json = nlohmann::json;

// Global config
std::string OPENWEBUI_URL;
std::string API_KEY;
std::string MODEL_NAME;
std::string KNOWLEDGE_BASE_IDS;
std::string PROMPT;
std::string TRIGGER_START;
std::string TRIGGER_STOP;

std::mutex analysis_mutex;
std::atomic<int> counter_value{0};
std::atomic<bool> in_analysis{false};

// Simple INI parser
std::map<std::string, std::string> parse_ini(const std::string& filename) {
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    std::string line, section;

    while (std::getline(file, line)) {
        // Remove comments
        size_t comment_pos = line.find_first_of(";#");
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
        } else {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                key.erase(0, key.find_first_not_of(" \t\r\n"));
                key.erase(key.find_last_not_of(" \t\r\n") + 1);
                value.erase(0, value.find_first_not_of(" \t\r\n"));
                value.erase(value.find_last_not_of(" \t\r\n") + 1);
                config[section + "." + key] = value;
            }
        }
    }

    return config;
}

// Load config
bool load_config(const std::string& path) {
    auto config = parse_ini(path);
    OPENWEBUI_URL = config["openai.base_url"];
    API_KEY = config["openai.api_key"];
    MODEL_NAME = config["openai.model_name"];
    KNOWLEDGE_BASE_IDS = config["analysis.knowledge_base_ids"];
    PROMPT = config["prompts.prompt"];
    TRIGGER_START = config["triggers.start"];
    std::transform(TRIGGER_START.begin(), TRIGGER_START.end(), TRIGGER_START.begin(), ::tolower);
    TRIGGER_STOP = config["triggers.stop"];
    std::transform(TRIGGER_STOP.begin(), TRIGGER_STOP.end(), TRIGGER_STOP.begin(), ::tolower);
    return !(OPENWEBUI_URL.empty() || API_KEY.empty());
}

// Substring check
bool contains_substring(const std::string& str, const std::string& sub) {
    return sub.empty() || str.find(sub) != std::string::npos;
}

// AI analysis
void analyze_text(const std::string& text) {
    in_analysis = true;
    int analysis_id = ++counter_value;
    std::cout << "Analysis of Recording[" << analysis_id << "] Started ------------------->>>\n";

    std::string filename = "results_analysis" + std::to_string(analysis_id) + ".txt";
    std::ofstream file(filename);
    file << "Using model: " << MODEL_NAME << "\n";
    file << "Prompt: " << PROMPT << "\n" << text << "\n";

    // Set the OpenWebUI Interface
    try {
            openai::start({
                API_KEY
            });

    // Create the request body
        json body = {
            {"model", MODEL_NAME},
            {"messages", {{{"role", "user"}, {"content", PROMPT + "\n" + text}}}},
            {"stream", false},
            {"knowledge_base_ids", {KNOWLEDGE_BASE_IDS}},
            {"enable_websearch", true}
        };

        // Call OpenWebUI
        auto chat = openai::chat().create(body);
        std::string response_string = chat["choices"][0]["message"]["content"];
   
        file << "\n\nFull response received:\n" << response_string << "\n";
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id << "] failed: " << e.what() << "\n";
    }

    std::cout << "Analysis of Recording[" << analysis_id << "] Finished ------------------->>>\n";
    in_analysis = false;
}

// Main loop
int main() {
    if (!load_config("./config.ini")) {
        std::cerr << "Failed to load config.ini\n";
        return 1;
    }

    std::cout << "Listening for input...\n";

    std::string line;
    std::string collected_text;
    bool collect_text = false;

    while (std::getline(std::cin, line)) {
        std::cout << line << "\n";

        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);

        if (contains_substring(lower_line, TRIGGER_START)) {
            if (collect_text) {
                std::cout << "Recording has already been started ------------------->>>\n";
            } else {
                std::cout << "Recording started ------------------->>>\n";
                collected_text.clear();
                collect_text = true;
            }
        }

        if (contains_substring(lower_line, TRIGGER_STOP)) {
            if (!collect_text) {
                std::cout << "No recording is currently running ------------------->>>\n";
            } else if (in_analysis) {
                std::cout << "A previous recording is still being analyzed ------------------->>>\n";
            } else {
                std::cout << "Recording stopped ------------------->>>\n";
                collected_text += line + "\n";
                std::thread(analyze_text, collected_text).detach();
                collected_text.clear();
                collect_text = false;
            }
        }

        if (collect_text) {
            collected_text += line + "\n";
        }
    }

    return 0;
}

