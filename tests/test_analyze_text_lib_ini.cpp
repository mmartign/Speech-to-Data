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

#include <sstream>

#include "../analyze_text_lib_ini.h"

TEST_CASE("parse_ini_stream parses sections, comments, and whitespace", "[ini]") {
    std::istringstream input(
        "[openai]\n"
        "base_url = http://localhost:8080/api\n"
        "; a full-line comment\n"
        "api_key=sk-test   # trailing comment\n"
        "\n"
        "[triggers]\n"
        "start = Start recording\n");

    const auto config = parse_ini_stream(input);
    REQUIRE(config.size() == 3);
    CHECK(config.at("openai.base_url") == "http://localhost:8080/api");
    CHECK(config.at("openai.api_key") == "sk-test");
    CHECK(config.at("triggers.start") == "Start recording");
}

TEST_CASE("parse_ini_stream ignores keys outside any section only by prefixing empty section", "[ini]") {
    std::istringstream input("orphan_key = value\n");
    const auto config = parse_ini_stream(input);
    REQUIRE(config.size() == 1);
    CHECK(config.at(".orphan_key") == "value");
}

TEST_CASE("parse_ini_stream lets a later duplicate key win", "[ini]") {
    std::istringstream input(
        "[s]\n"
        "key = first\n"
        "key = second\n");
    const auto config = parse_ini_stream(input);
    CHECK(config.at("s.key") == "second");
}
