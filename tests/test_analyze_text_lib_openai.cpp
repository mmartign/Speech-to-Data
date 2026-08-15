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
#include <catch2/catch_test_macros.hpp>

#include "../analyze_text_lib_openai.h"

TEST_CASE("extract_message_content reads a plain string content field", "[openai]") {
    const json response = {
        {"choices", json::array({
            {{"message", {{"role", "assistant"}, {"content", "  hello there  "}}}}
        })}
    };
    CHECK(extract_message_content(response) == "hello there");
}

TEST_CASE("extract_message_content joins array content parts with newlines", "[openai]") {
    const json response = {
        {"choices", json::array({
            {{"message", {{"role", "assistant"}, {"content", json::array({
                json{{"type", "text"}, {"text", "first part"}},
                "second part"
            })}}}}
        })}
    };
    CHECK(extract_message_content(response) == "first part\nsecond part");
}

TEST_CASE("extract_message_content returns empty string for malformed responses", "[openai]") {
    CHECK(extract_message_content(json::object()).empty());
    CHECK(extract_message_content(json{{"choices", json::array()}}).empty());
}

TEST_CASE("extract_api_error reads a plain string error field", "[openai]") {
    const json response = {{"error", "something broke"}};
    CHECK(extract_api_error(response) == "something broke");
}

TEST_CASE("extract_api_error dumps a structured error field", "[openai]") {
    const json error_obj = {{"message", "detail"}};
    const json response = {{"error", error_obj}};
    CHECK(extract_api_error(response) == error_obj.dump());
}

TEST_CASE("extract_api_error falls back to detail then message", "[openai]") {
    CHECK(extract_api_error(json{{"detail", "bad request"}}) == "bad request");
    CHECK(extract_api_error(json{{"message", "oops"}}) == "oops");
    CHECK(extract_api_error(json::object()).empty());
    CHECK(extract_api_error(json::array()).empty());
}
