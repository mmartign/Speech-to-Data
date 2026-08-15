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

#include "../analyze_text_lib_text.h"

TEST_CASE("trim_whitespace strips leading/trailing whitespace only", "[text]") {
    CHECK(trim_whitespace("  hello world  ") == "hello world");
    CHECK(trim_whitespace("\t\r\nhello\n") == "hello");
    CHECK(trim_whitespace("no-op") == "no-op");
    CHECK(trim_whitespace("   ") == "");
    CHECK(trim_whitespace("") == "");
}

TEST_CASE("strip_trailing_newlines removes only trailing CR/LF", "[text]") {
    CHECK(strip_trailing_newlines("line one\n\n") == "line one");
    CHECK(strip_trailing_newlines("line one\r\n") == "line one");
    CHECK(strip_trailing_newlines("no newline") == "no newline");
    CHECK(strip_trailing_newlines("\nleading kept\n") == "\nleading kept");
}

TEST_CASE("ensure_trailing_slash appends exactly one slash", "[text]") {
    CHECK(ensure_trailing_slash("http://host/api") == "http://host/api/");
    CHECK(ensure_trailing_slash("http://host/api/") == "http://host/api/");
    CHECK(ensure_trailing_slash("") == "");
}

TEST_CASE("split_config_list trims items and drops empties", "[text]") {
    const auto items = split_config_list("a, b ,,c ,");
    REQUIRE(items.size() == 3);
    CHECK(items[0] == "a");
    CHECK(items[1] == "b");
    CHECK(items[2] == "c");
}

TEST_CASE("split_config_list on empty string yields no items", "[text]") {
    CHECK(split_config_list("").empty());
    CHECK(split_config_list("   ").empty());
}

TEST_CASE("escape_for_single_quotes closes/reopens quotes around embedded quotes", "[text]") {
    CHECK(escape_for_single_quotes("plain") == "plain");
    // A single embedded quote becomes: close-quote, escaped-quote, reopen-quote.
    const std::string expected = std::string("it") + "'" + "\\" + "'" + "'" + "s";
    CHECK(escape_for_single_quotes("it's") == expected);
}

TEST_CASE("contains_substring treats empty needle as always matching", "[text]") {
    CHECK(contains_substring("hello world", ""));
    CHECK(contains_substring("hello world", "world"));
    CHECK_FALSE(contains_substring("hello world", "xyz"));
}
