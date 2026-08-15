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

#include "../analyze_text_lib_cleanup.h"

TEST_CASE("strip_json_like_comments removes // and block comments outside strings", "[cleanup]") {
    const std::string input =
        "{\n"
        "  // a line comment\n"
        "  \"a\": 1, /* a block\n"
        "  comment */ \"b\": \"// not a comment\"\n"
        "}\n";
    const std::string result = strip_json_like_comments(input);
    CHECK(result.find("// a line comment") == std::string::npos);
    CHECK(result.find("a block") == std::string::npos);
    // Comment-like text inside a real string literal must survive.
    CHECK(result.find("\"// not a comment\"") != std::string::npos);
}

TEST_CASE("strip_internal_reasoning_tags removes a full thought block", "[cleanup]") {
    const std::string input = "<unused1>internal thought stuff<unused2>visible text";
    CHECK(strip_internal_reasoning_tags(input) == "visible text");
}

TEST_CASE("strip_internal_reasoning_tags leaves plain text untouched", "[cleanup]") {
    CHECK(strip_internal_reasoning_tags("  Just plain text.  ") == "Just plain text.");
}

TEST_CASE("strip_internal_reasoning_tags strips a bare stray tag without a thought marker", "[cleanup]") {
    CHECK(strip_internal_reasoning_tags("<unused5>plain answer") == "plain answer");
}

TEST_CASE("strip_internal_reasoning_tags drops everything after an unterminated thought tag", "[cleanup]") {
    // No closing <unused...> tag exists, so the opening tag and everything after it is dropped.
    CHECK(strip_internal_reasoning_tags("<unused1>thought with no closing tag") == "");
}
