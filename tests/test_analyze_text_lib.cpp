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

#include "../analyze_text_lib.h"

// ── Text / config helpers ───────────────────────────────────────────────────

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

// ── INI parsing ──────────────────────────────────────────────────────────────

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

// ── Model-response text cleanup ─────────────────────────────────────────────

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

// ── FHIR bundle extraction ──────────────────────────────────────────────────

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

// ── OpenAI-style response parsing ───────────────────────────────────────────

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
