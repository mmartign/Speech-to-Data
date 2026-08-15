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
#include "analyze_text_lib_cleanup.h"

#include <cctype>

#include "analyze_text_lib_text.h"

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
