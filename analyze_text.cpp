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
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <utility>
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
std::mutex tts_mutex;

class AnalysisSession {
public:
    AnalysisSession(std::mutex& mutex, std::atomic<bool>& state)
        : lock_(mutex), state_(state) {}

    ~AnalysisSession() {
        state_ = false;
    }

private:
    std::unique_lock<std::mutex> lock_;
    std::atomic<bool>& state_;
};

std::string strip_trailing_newlines(std::string text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

std::string escape_for_quotes(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char c : text) {
        if (c == '\\' || c == '\"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return escaped;
}

void speak_text(const std::string& text) {
    const std::string trimmed = strip_trailing_newlines(text);
    if (trimmed.empty()) {
        return;
    }

    const std::string escaped = escape_for_quotes(trimmed);
    const std::string cmd = "say -v Alex \"" + escaped + "\" >/dev/null 2>&1 &";

    std::lock_guard<std::mutex> lock(tts_mutex);
    std::system(cmd.c_str());
}

void say_info(const std::string& message) {
    std::cout << message;
    speak_text(message);
}

void say_error(const std::string& message) {
    std::cerr << message;
    speak_text(message);
}

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
    std::ifstream file_check(path);
    if (!file_check.is_open()) {
        say_error("Unable to open config file: " + path + "\n");
        return false;
    }
    file_check.close();

    auto config = parse_ini(path);

    std::vector<std::string> missing_keys;
    auto require_value = [&](const std::string& key, std::string& destination) {
        auto it = config.find(key);
        if (it == config.end() || it->second.empty()) {
            missing_keys.push_back(key);
            return;
        }
        destination = it->second;
    };

    require_value("openai.base_url", OPENWEBUI_URL);
    require_value("openai.api_key", API_KEY);
    require_value("openai.model_name", MODEL_NAME);
    require_value("prompts.prompt", PROMPT);
    require_value("triggers.start", TRIGGER_START);
    require_value("triggers.stop", TRIGGER_STOP);

    auto kb_it = config.find("analysis.knowledge_base_ids");
    KNOWLEDGE_BASE_IDS = (kb_it != config.end()) ? kb_it->second : std::string{};

    if (!missing_keys.empty()) {
        std::ostringstream oss;
        oss << "Missing required config values:";
        for (const auto& key : missing_keys) {
            oss << ' ' << key;
        }
        oss << "\n";
        say_error(oss.str());
        return false;
    }

    std::transform(TRIGGER_START.begin(), TRIGGER_START.end(), TRIGGER_START.begin(), ::tolower);
    std::transform(TRIGGER_STOP.begin(), TRIGGER_STOP.end(), TRIGGER_STOP.begin(), ::tolower);

    if (KNOWLEDGE_BASE_IDS.empty()) {
        say_error("Warning: analysis.knowledge_base_ids is not set; knowledge base lookups will be skipped.\n");
    }

    return true;
}

// Substring check
bool contains_substring(const std::string& str, const std::string& sub) {
    return sub.empty() || str.find(sub) != std::string::npos;
}

// AI analysis with fresh context for each request
void analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex, in_analysis);
    const int analysis_id = ++counter_value;
    say_info("Analysis of Recording[" + std::to_string(analysis_id) + "] Started ------------------->>>\n");

    const std::string filename = "results_analysis" + std::to_string(analysis_id) + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error("[ERROR] Unable to open results file: " + filename + "\n");
        say_info("Analysis of Recording[" + std::to_string(analysis_id) + "] Finished ------------------->>>\n");
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error("[ERROR] Failed to write analysis header to " + filename + "\n");
    }

    try {
        openai::start({
            API_KEY
        });

        json body = {
            {"model", MODEL_NAME},
            {"messages", {
                {{"role", "system"}, {"content", "You are a helpful assistant."}},
                {{"role", "user"}, {"content", PROMPT + "\n" + text}}
            }},
            {"stream", false},
            {"enable_websearch", true}
        };

        if (!KNOWLEDGE_BASE_IDS.empty()) {
            body["knowledge_base_ids"] = json::array({KNOWLEDGE_BASE_IDS});
        }

        auto chat = openai::chat().create(body);
        const std::string response_string = chat["choices"][0]["message"]["content"].get<std::string>();

        file << "\n\nFull response received:\n" << response_string << "\n";
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id << "] failed: " << e.what() << "\n";
        say_error(std::string{"[ERROR] Analysis["} + std::to_string(analysis_id) + "] failed: " + e.what() + "\n");
    }

    if (!file) {
        say_error("[ERROR] Writing to results file failed for Analysis[" + std::to_string(analysis_id) + "]\n");
    }

    say_info("Analysis of Recording[" + std::to_string(analysis_id) + "] Finished ------------------->>>\n");
}

// Main loop
int main() {
    if (!load_config("./config.ini")) {
        say_error("Failed to load config.ini\n");
        return 1;
    }

    say_info("Listening for input...\n");

    std::string line;
    std::string collected_text;
    bool collect_text = false;

    while (std::getline(std::cin, line)) {
        std::cout << line << std::endl;

        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);

        const bool line_contains_start = contains_substring(lower_line, TRIGGER_START);
        const bool line_contains_stop = contains_substring(lower_line, TRIGGER_STOP);

        if (line_contains_start) {
            if (collect_text) {
                say_info("Recording has already been started ------------------->>>\n");
            } else {
                say_info("Recording started ------------------->>>\n");
                collected_text.clear();
                collect_text = true;
            }
        }

        if (line_contains_stop) {
            if (!collect_text) {
                say_info("No recording is currently running ------------------->>>\n");
            } else if (in_analysis) {
                say_info("A previous recording is still being analyzed ------------------->>>\n");
            } else {
                say_info("Recording stopped ------------------->>>\n");
                std::string text_to_analyze = collected_text;
                collected_text.clear();
                collect_text = false;
                in_analysis = true;
                std::thread(analyze_text, std::move(text_to_analyze)).detach();
            }
        }

        if (collect_text && !line_contains_start && !line_contains_stop) {
            collected_text += line + "\n";
        }
    }

    return 0;
}
