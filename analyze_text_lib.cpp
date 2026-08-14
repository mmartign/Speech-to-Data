// -*- coding: utf-8 -*-
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of the Spazio IT Speech-to-Knowledge project.
//
// Copyright (C) 2025-2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
#include "analyze_text_lib.h"

#include <cctype>
#include <fstream>
#include <sstream>

std::string strip_trailing_newlines(std::string text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

std::string trim_whitespace(std::string text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string ensure_trailing_slash(std::string url) {
    if (!url.empty() && url.back() != '/') {
        url.push_back('/');
    }
    return url;
}

std::vector<std::string> split_config_list(const std::string& value) {
    std::vector<std::string> items;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_whitespace(item);
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

std::string escape_for_single_quotes(const std::string& text) {
    // POSIX-shell escaping for single-quoted command arguments.
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (char c : text) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(c);
        }
    }
    return escaped;
}

bool contains_substring(const std::string& str, const std::string& sub) {
    return sub.empty() || str.find(sub) != std::string::npos;
}

std::map<std::string, std::string> parse_ini_stream(std::istream& input) {
    std::map<std::string, std::string> config;
    std::string line, section;

    while (std::getline(input, line)) {
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

std::map<std::string, std::string> parse_ini(const std::string& filename) {
    std::ifstream file(filename);
    return parse_ini_stream(file);
}

std::string strip_json_like_comments(const std::string& input) {
    // Supports model responses that include JSON with JS-style comments.
    std::string out;
    out.reserve(input.size());

    bool in_string = false;
    bool escaped = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (size_t i = 0; i < input.size(); ++i) {
        const char c = input[i];
        const char next = (i + 1 < input.size()) ? input[i + 1] : '\0';

        if (in_line_comment) {
            if (c == '\n') {
                in_line_comment = false;
                out.push_back(c);
            }
            continue;
        }

        if (in_block_comment) {
            if (c == '*' && next == '/') {
                in_block_comment = false;
                ++i;
                continue;
            }
            if (c == '\n') {
                out.push_back(c);
            }
            continue;
        }

        if (in_string) {
            out.push_back(c);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            out.push_back(c);
            continue;
        }

        if (c == '/' && next == '/') {
            in_line_comment = true;
            ++i;
            continue;
        }

        if (c == '/' && next == '*') {
            in_block_comment = true;
            ++i;
            continue;
        }

        out.push_back(c);
    }

    return out;
}

bool starts_with_unused_tag_at(const std::string& text, size_t pos) {
    constexpr const char* prefix = "<unused";
    constexpr size_t prefix_len = 7;
    if (pos + prefix_len >= text.size() || text.compare(pos, prefix_len, prefix) != 0) {
        return false;
    }
    size_t i = pos + prefix_len;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    return i < text.size() && text[i] == '>';
}

std::string strip_internal_reasoning_tags(std::string text) {
    // Remove leaked internal <unused...> segments before user-facing output.
    size_t cursor = 0;
    while (cursor < text.size()) {
        size_t tag_pos = text.find("<unused", cursor);
        if (tag_pos == std::string::npos) {
            break;
        }
        if (!starts_with_unused_tag_at(text, tag_pos)) {
            cursor = tag_pos + 1;
            continue;
        }

        size_t thought_pos = text.find("thought", tag_pos);
        if (thought_pos == std::string::npos || thought_pos > tag_pos + 48) {
            cursor = tag_pos + 1;
            continue;
        }

        const size_t next_tag = text.find("<unused", thought_pos);
        if (next_tag == std::string::npos || !starts_with_unused_tag_at(text, next_tag)) {
            text.erase(tag_pos);
            break;
        }

        text.erase(tag_pos, next_tag - tag_pos);
        cursor = tag_pos;
    }

    while (true) {
        size_t tag_pos = text.find("<unused");
        if (tag_pos == std::string::npos || !starts_with_unused_tag_at(text, tag_pos)) {
            break;
        }
        size_t end = text.find('>', tag_pos);
        if (end == std::string::npos) {
            break;
        }
        text.erase(tag_pos, (end - tag_pos) + 1);
    }

    return trim_whitespace(text);
}

bool is_fhir_bundle_object(const json& candidate) {
    if (!candidate.is_object()) {
        return false;
    }
    const auto type_it = candidate.find("resourceType");
    if (type_it == candidate.end() || !type_it->is_string()) {
        return false;
    }
    return type_it->get<std::string>() == "Bundle";
}

bool extract_fhir_bundle_from_text(const std::string& text, json& bundle, size_t& start_pos, size_t& end_pos) {
    bool in_string = false;
    bool escaped = false;

    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];

        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }

        if (c != '{') {
            continue;
        }

        size_t depth = 0;
        bool local_in_string = false;
        bool local_escaped = false;
        bool completed = false;

        for (size_t j = i; j < text.size(); ++j) {
            const char cj = text[j];
            if (local_in_string) {
                if (local_escaped) {
                    local_escaped = false;
                } else if (cj == '\\') {
                    local_escaped = true;
                } else if (cj == '"') {
                    local_in_string = false;
                }
                continue;
            }

            if (cj == '"') {
                local_in_string = true;
                continue;
            }
            if (cj == '{') {
                ++depth;
            } else if (cj == '}') {
                if (depth == 0) {
                    break;
                }
                --depth;
                if (depth == 0) {
                    const std::string candidate = text.substr(i, j - i + 1);
                    try {
                        json parsed;
                        try {
                            parsed = json::parse(candidate);
                        } catch (...) {
                            parsed = json::parse(strip_json_like_comments(candidate));
                        }
                        if (is_fhir_bundle_object(parsed)) {
                            // Return the first valid Bundle object found in the response text.
                            bundle = std::move(parsed);
                            start_pos = i;
                            end_pos = j + 1;
                            return true;
                        }
                    } catch (...) {
                        // Keep searching for the next candidate JSON object.
                    }
                    completed = true;
                    break;
                }
            }
        }

        if (!completed) {
            break;
        }
    }

    return false;
}

bool extract_revised_bundle(const json& mapper_output, json& revised_bundle) {
    // Accept either raw Bundle output or wrapped mapper payload.
    if (is_fhir_bundle_object(mapper_output)) {
        revised_bundle = mapper_output;
        return true;
    }

    const auto accepted_it = mapper_output.find("acceptedBundle");
    if (accepted_it != mapper_output.end() && is_fhir_bundle_object(*accepted_it)) {
        revised_bundle = *accepted_it;
        return true;
    }

    return false;
}

// Safely extract a textual message content from an OpenAI-style response
std::string extract_message_content(const json& response) {
    // Normalize multiple OpenAI response shapes into plain text.
    const auto choices_it = response.find("choices");
    if (choices_it == response.end() || !choices_it->is_array() || choices_it->empty()) {
        return {};
    }

    const auto& first_choice = (*choices_it)[0];
    if (!first_choice.is_object()) {
        return {};
    }

    const auto message_it = first_choice.find("message");
    if (message_it == first_choice.end() || !message_it->is_object()) {
        return {};
    }

    const auto content_it = message_it->find("content");
    if (content_it == message_it->end()) {
        return {};
    }

    if (content_it->is_string()) {
        return trim_whitespace(content_it->get<std::string>());
    }

    if (content_it->is_array()) {
        std::string combined;
        for (const auto& part : *content_it) {
            std::string part_text;
            if (part.is_string()) {
                part_text = part.get<std::string>();
            } else if (part.is_object()) {
                const auto text_it = part.find("text");
                if (text_it != part.end() && text_it->is_string()) {
                    part_text = text_it->get<std::string>();
                } else {
                    const auto content_it2 = part.find("content");
                    if (content_it2 != part.end() && content_it2->is_string()) {
                        part_text = content_it2->get<std::string>();
                    }
                }
            }
            if (part_text.empty()) {
                continue;
            }
            if (!combined.empty()) {
                combined.push_back('\n');
            }
            combined += part_text;
        }
        return trim_whitespace(combined);
    }

    return {};
}

std::string extract_api_error(const json& response) {
    if (!response.is_object()) {
        return {};
    }

    const auto error_it = response.find("error");
    if (error_it != response.end()) {
        if (error_it->is_string()) {
            return error_it->get<std::string>();
        }
        return error_it->dump();
    }

    const auto detail_it = response.find("detail");
    if (detail_it != response.end()) {
        if (detail_it->is_string()) {
            return detail_it->get<std::string>();
        }
        return detail_it->dump();
    }

    const auto message_it = response.find("message");
    if (message_it != response.end() && message_it->is_string()) {
        return message_it->get<std::string>();
    }

    return {};
}
