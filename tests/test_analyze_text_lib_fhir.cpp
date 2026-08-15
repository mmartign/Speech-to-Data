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

#include "../analyze_text_lib_fhir.h"

TEST_CASE("is_fhir_bundle_object requires resourceType == Bundle", "[fhir]") {
    CHECK(is_fhir_bundle_object(json{{"resourceType", "Bundle"}}));
    CHECK_FALSE(is_fhir_bundle_object(json{{"resourceType", "Patient"}}));
    CHECK_FALSE(is_fhir_bundle_object(json{{"other", "field"}}));
    CHECK_FALSE(is_fhir_bundle_object(json::array()));
}

TEST_CASE("extract_fhir_bundle_from_text finds a bundle embedded in narrative text", "[fhir]") {
    const std::string text =
        "Here is the result: "
        "{\"resourceType\":\"Bundle\",\"type\":\"transaction\",\"entry\":[]} "
        "Thanks.";
    json bundle;
    size_t start = 0, end = 0;
    REQUIRE(extract_fhir_bundle_from_text(text, bundle, start, end));
    CHECK(bundle["resourceType"] == "Bundle");
    CHECK(text.substr(start, end - start) ==
          "{\"resourceType\":\"Bundle\",\"type\":\"transaction\",\"entry\":[]}");
}

TEST_CASE("extract_fhir_bundle_from_text tolerates JS-style comments via fallback parse", "[fhir]") {
    const std::string text =
        "{\n"
        "  // leading comment\n"
        "  \"resourceType\": \"Bundle\",\n"
        "  \"entry\": []\n"
        "}\n";
    json bundle;
    size_t start = 0, end = 0;
    REQUIRE(extract_fhir_bundle_from_text(text, bundle, start, end));
    CHECK(bundle["resourceType"] == "Bundle");
}

TEST_CASE("extract_fhir_bundle_from_text ignores braces inside string values", "[fhir]") {
    const std::string text =
        "{\"resourceType\":\"Bundle\",\"note\":\"a { curly brace } inside a string\"}";
    json bundle;
    size_t start = 0, end = 0;
    REQUIRE(extract_fhir_bundle_from_text(text, bundle, start, end));
    CHECK(bundle["resourceType"] == "Bundle");
    CHECK(end == text.size());
}

TEST_CASE("extract_fhir_bundle_from_text returns false when no Bundle is present", "[fhir]") {
    const std::string text = "Just some prose with no JSON at all.";
    json bundle;
    size_t start = 0, end = 0;
    CHECK_FALSE(extract_fhir_bundle_from_text(text, bundle, start, end));

    const std::string non_bundle_json = "{\"resourceType\":\"Patient\",\"id\":\"1\"}";
    CHECK_FALSE(extract_fhir_bundle_from_text(non_bundle_json, bundle, start, end));
}

TEST_CASE("extract_revised_bundle accepts a raw Bundle", "[fhir]") {
    const json input = {{"resourceType", "Bundle"}, {"entry", json::array()}};
    json revised;
    REQUIRE(extract_revised_bundle(input, revised));
    CHECK(revised["resourceType"] == "Bundle");
}

TEST_CASE("extract_revised_bundle unwraps an acceptedBundle field", "[fhir]") {
    const json input = {
        {"acceptedBundle", {{"resourceType", "Bundle"}, {"entry", json::array()}}},
        {"rejectedBundle", {{"resourceType", "Bundle"}, {"entry", json::array()}}},
    };
    json revised;
    REQUIRE(extract_revised_bundle(input, revised));
    CHECK(revised["resourceType"] == "Bundle");
}

TEST_CASE("extract_revised_bundle fails when neither shape is present", "[fhir]") {
    const json input = {{"outcome", "no bundle here"}};
    json revised;
    CHECK_FALSE(extract_revised_bundle(input, revised));
}
