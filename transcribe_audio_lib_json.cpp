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
#include "transcribe_audio_lib_json.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

size_t skip_json_whitespace(const std::string& json, size_t pos) {
    while (pos < json.size()) {
        const unsigned char c = static_cast<unsigned char>(json[pos]);
        if (!std::isspace(c)) {
            break;
        }
        ++pos;
    }
    return pos;
}

std::optional<std::string> parse_json_quoted_string(const std::string& json, size_t& pos) {
    if (pos >= json.size() || json[pos] != '"') {
        return std::nullopt;
    }
    ++pos;
    std::string out;
    out.reserve(16);
    while (pos < json.size()) {
        const char c = json[pos++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (pos >= json.size()) {
                return std::nullopt;
            }
            const char esc = json[pos++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                // Keep unicode escapes literal: protocol routing fields are
                // ASCII names and ids, so full Unicode decoding is unnecessary.
                case 'u': out += "\\u"; break;
                default: out.push_back(esc); break;
            }
            continue;
        }
        out.push_back(c);
    }
    return std::nullopt;
}

std::string extract_json_string_field(const std::string& json, const std::string& field) {
    // This parser is intentionally shallow because start/stop frames are flat
    // JSON objects controlled by our mobile client.
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return {};
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return {};
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    auto parsed = parse_json_quoted_string(json, value_pos);
    return parsed.has_value() ? *parsed : std::string{};
}

int extract_json_int_field(const std::string& json, const std::string& field, int fallback) {
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return fallback;
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    size_t end = value_pos;
    if (end < json.size() && (json[end] == '-' || json[end] == '+')) {
        ++end;
    }
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) {
        ++end;
    }
    if (end == value_pos || (end == value_pos + 1 && (json[value_pos] == '-' || json[value_pos] == '+'))) {
        return fallback;
    }
    try {
        return std::stoi(json.substr(value_pos, end - value_pos));
    } catch (...) {
        return fallback;
    }
}

std::string ascii_lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim_ascii_whitespace(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto first = std::find_if(s.begin(), s.end(), not_space);
    if (first == s.end()) {
        return {};
    }
    const auto last = std::find_if(s.rbegin(), s.rend(), not_space).base();
    return std::string(first, last);
}

bool json_value_delimiter(char c) {
    return c == ',' || c == '}' || c == ']' || std::isspace(static_cast<unsigned char>(c));
}

bool extract_json_bool_field(const std::string& json, const std::string& field, bool fallback) {
    // Accept both real booleans and string-ish form values to keep the protocol
    // tolerant of clients that serialize UI settings as strings.
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return fallback;
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    if (value_pos >= json.size()) {
        return fallback;
    }

    if (json[value_pos] == '"') {
        auto parsed = parse_json_quoted_string(json, value_pos);
        if (!parsed.has_value()) {
            return fallback;
        }
        const std::string value = ascii_lower_copy(trim_ascii_whitespace(*parsed));
        if (value == "true" || value == "1" || value == "on" || value == "yes") {
            return true;
        }
        if (value == "false" || value == "0" || value == "off" || value == "no") {
            return false;
        }
        return fallback;
    }

    const size_t remaining = json.size() - value_pos;
    if (remaining >= 4 &&
        json.compare(value_pos, 4, "true") == 0 &&
        (value_pos + 4 == json.size() || json_value_delimiter(json[value_pos + 4]))) {
        return true;
    }
    if (remaining >= 5 &&
        json.compare(value_pos, 5, "false") == 0 &&
        (value_pos + 5 == json.size() || json_value_delimiter(json[value_pos + 5]))) {
        return false;
    }
    if (json[value_pos] == '1' &&
        (value_pos + 1 == json.size() || json_value_delimiter(json[value_pos + 1]))) {
        return true;
    }
    if (json[value_pos] == '0' &&
        (value_pos + 1 == json.size() || json_value_delimiter(json[value_pos + 1]))) {
        return false;
    }
    return fallback;
}

bool json_message_type_is(const std::string& json, const char* expected) {
    return extract_json_string_field(json, "type") == expected;
}
