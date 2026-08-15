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

#include "../transcribe_audio_lib_json.h"

TEST_CASE("json_escape escapes control characters and quotes", "[json]") {
    CHECK(json_escape("plain") == "plain");
    CHECK(json_escape("a\"b") == "a\\\"b");
    CHECK(json_escape("a\\b") == "a\\\\b");
    CHECK(json_escape("line1\nline2") == "line1\\nline2");
    CHECK(json_escape(std::string(1, '\x01')) == "\\u0001");
}

TEST_CASE("skip_json_whitespace advances past spaces and stops at content", "[json]") {
    CHECK(skip_json_whitespace("   x", 0) == 3);
    CHECK(skip_json_whitespace("x", 0) == 0);
    CHECK(skip_json_whitespace("   ", 0) == 3);
}

TEST_CASE("parse_json_quoted_string parses simple and escaped strings", "[json]") {
    size_t pos = 0;
    auto result = parse_json_quoted_string("\"hello\"", pos);
    REQUIRE(result.has_value());
    CHECK(*result == "hello");

    pos = 0;
    result = parse_json_quoted_string(R"("a\"b")", pos);
    REQUIRE(result.has_value());
    CHECK(*result == "a\"b");

    pos = 0;
    CHECK_FALSE(parse_json_quoted_string("no quotes here", pos).has_value());

    pos = 0;
    CHECK_FALSE(parse_json_quoted_string("\"unterminated", pos).has_value());
}

TEST_CASE("extract_json_string_field finds a flat string field", "[json]") {
    const std::string msg = R"({"type":"start","sessionId":"abc123"})";
    CHECK(extract_json_string_field(msg, "type") == "start");
    CHECK(extract_json_string_field(msg, "sessionId") == "abc123");
    CHECK(extract_json_string_field(msg, "missing").empty());
}

TEST_CASE("extract_json_int_field parses integers and falls back otherwise", "[json]") {
    const std::string msg = R"({"sampleRate":16000,"channels":-1})";
    CHECK(extract_json_int_field(msg, "sampleRate", -1) == 16000);
    CHECK(extract_json_int_field(msg, "channels", 0) == -1);
    CHECK(extract_json_int_field(msg, "missing", 42) == 42);
}

TEST_CASE("ascii_lower_copy lowercases ASCII letters only", "[json]") {
    CHECK(ascii_lower_copy("MiXeD Case 123") == "mixed case 123");
}

TEST_CASE("json_value_delimiter recognizes JSON value terminators", "[json]") {
    CHECK(json_value_delimiter(','));
    CHECK(json_value_delimiter('}'));
    CHECK(json_value_delimiter(']'));
    CHECK(json_value_delimiter(' '));
    CHECK_FALSE(json_value_delimiter('a'));
}

TEST_CASE("extract_json_bool_field accepts real booleans, numerics, and string forms", "[json]") {
    CHECK(extract_json_bool_field(R"({"timestamp":true})", "timestamp", false));
    CHECK_FALSE(extract_json_bool_field(R"({"timestamp":false})", "timestamp", true));
    CHECK(extract_json_bool_field(R"({"timestamp":1})", "timestamp", false));
    CHECK(extract_json_bool_field(R"({"timestamp":"yes"})", "timestamp", false));
    CHECK_FALSE(extract_json_bool_field(R"({"timestamp":"off"})", "timestamp", true));
    CHECK(extract_json_bool_field(R"({})", "missing", true));
}

TEST_CASE("json_message_type_is matches the type field", "[json]") {
    const std::string msg = R"({"type":"stop"})";
    CHECK(json_message_type_is(msg, "stop"));
    CHECK_FALSE(json_message_type_is(msg, "start"));
}
