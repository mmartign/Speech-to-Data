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

#include <regex>

#include "../transcribe_audio_lib_text.h"

TEST_CASE("trim removes standard whitespace characters from both ends", "[text]") {
    CHECK(trim("  hello  ") == "hello");
    CHECK(trim("\t\nhello\f\v") == "hello");
    CHECK(trim("") == "");
    CHECK(trim("   ") == "");
}

TEST_CASE("is_whisper_noise_token detects exact bracket/paren tokens", "[noise]") {
    CHECK(is_whisper_noise_token("[BLANK_AUDIO]"));
    CHECK(is_whisper_noise_token("(silence)"));
    CHECK(is_whisper_noise_token("[MUSIC]"));
    CHECK_FALSE(is_whisper_noise_token("[BLANK AUDIO]"));
}

TEST_CASE("is_whisper_noise_token detects hallucination phrases regardless of case/punctuation", "[noise]") {
    CHECK(is_whisper_noise_token("Thank you."));
    CHECK(is_whisper_noise_token("THANKS FOR WATCHING!"));
    CHECK(is_whisper_noise_token("Subscribe"));
    CHECK_FALSE(is_whisper_noise_token("The patient reports chest pain."));
    CHECK_FALSE(is_whisper_noise_token("Thank you for your patience while we review the results."));
}

TEST_CASE("format_datetime formats as [YYYY-MM-DD HH:MM:SS]", "[text]") {
    const auto now = std::chrono::system_clock::now();
    const std::string formatted = format_datetime(now);
    CHECK(std::regex_match(formatted, std::regex(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])")));
}
