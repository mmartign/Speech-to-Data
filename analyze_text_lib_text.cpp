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
#include "analyze_text_lib_text.h"

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
