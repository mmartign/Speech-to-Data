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
#pragma once

// Pure logic extracted from analyze_text.cpp: no network I/O, no shelling out,
// no dependency on the program's global config state. Kept separate so it can
// be linked into a unit-test binary without pulling in the OpenAI client or
// main().

#include <cstddef>
#include <istream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Text / config helpers ───────────────────────────────────────────────────
std::string strip_trailing_newlines(std::string text);
std::string trim_whitespace(std::string text);
std::string ensure_trailing_slash(std::string url);
std::vector<std::string> split_config_list(const std::string& value);
std::string escape_for_single_quotes(const std::string& text);
bool contains_substring(const std::string& str, const std::string& sub);

// ── INI parsing ──────────────────────────────────────────────────────────────
std::map<std::string, std::string> parse_ini_stream(std::istream& input);
std::map<std::string, std::string> parse_ini(const std::string& filename);

// ── Model-response text cleanup ─────────────────────────────────────────────
std::string strip_json_like_comments(const std::string& input);
bool starts_with_unused_tag_at(const std::string& text, size_t pos);
std::string strip_internal_reasoning_tags(std::string text);

// ── FHIR bundle extraction ──────────────────────────────────────────────────
bool is_fhir_bundle_object(const json& candidate);
bool extract_fhir_bundle_from_text(const std::string& text, json& bundle, size_t& start_pos, size_t& end_pos);
bool extract_revised_bundle(const json& mapper_output, json& revised_bundle);

// ── OpenAI-style response parsing ───────────────────────────────────────────
std::string extract_message_content(const json& response);
std::string extract_api_error(const json& response);
