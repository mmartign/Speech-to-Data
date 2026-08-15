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

#include "../transcribe_audio_lib_repetition.h"

namespace {
std::string join_words(const std::vector<std::string>& words) {
    std::string out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) out += ' ';
        out += words[i];
    }
    return out;
}
}

// ── Transcript word helpers ─────────────────────────────────────────────────

TEST_CASE("transcript_word_byte accepts alnum, apostrophe, and high bytes", "[words]") {
    CHECK(transcript_word_byte(static_cast<unsigned char>('a')));
    CHECK(transcript_word_byte(static_cast<unsigned char>('9')));
    CHECK(transcript_word_byte(static_cast<unsigned char>('\'')));
    CHECK(transcript_word_byte(static_cast<unsigned char>(0xE0)));
    CHECK_FALSE(transcript_word_byte(static_cast<unsigned char>(' ')));
    CHECK_FALSE(transcript_word_byte(static_cast<unsigned char>('.')));
}

TEST_CASE("normalize_transcript_word lowercases ASCII and strips edge apostrophes", "[words]") {
    CHECK(normalize_transcript_word("Hello") == "hello");
    CHECK(normalize_transcript_word("'twas") == "twas");
    CHECK(normalize_transcript_word("y'all'") == "y'all");
    CHECK(normalize_transcript_word("'''") == "");
}

TEST_CASE("extract_transcript_words splits on non-word bytes and records spans", "[words]") {
    const std::string text = "Hello, world!";
    const auto words = extract_transcript_words(text);
    REQUIRE(words.size() == 2);
    CHECK(words[0].normalized == "hello");
    CHECK(text.substr(words[0].begin, words[0].end - words[0].begin) == "Hello");
    CHECK(words[1].normalized == "world");
    CHECK(text.substr(words[1].begin, words[1].end - words[1].begin) == "world");
}

TEST_CASE("extract_transcript_words returns empty for text with no word bytes", "[words]") {
    CHECK(extract_transcript_words("   ...   ").empty());
    CHECK(extract_transcript_words("").empty());
}

// ── Repetition trimming ──────────────────────────────────────────────────────

TEST_CASE("find_repetitive_tail_trim_offset returns nullopt for short word lists", "[repetition]") {
    std::vector<TranscriptWordSpan> words(7);
    CHECK_FALSE(find_repetitive_tail_trim_offset(words).has_value());
}

TEST_CASE("trim_excessive_repetition leaves short repeats and normal text alone", "[repetition]") {
    const std::string short_repeat = "yes yes yes yes hello there";
    CHECK(trim_excessive_repetition(short_repeat) == short_repeat);

    const std::string plain = "the quick brown fox jumps over the lazy dog";
    CHECK(trim_excessive_repetition(plain) == plain);
}

TEST_CASE("trim_excessive_repetition trims a long repeated word run at the tail", "[repetition]") {
    std::vector<std::string> words = {"hello", "there"};
    for (int i = 0; i < 9; ++i) words.push_back("yes");
    const std::string input = join_words(words);
    const std::string expected = join_words({"hello", "there", "yes", "yes", "yes"});
    CHECK(trim_excessive_repetition(input) == expected);
}

TEST_CASE("trim_excessive_repetition trims a very long repeated run even mid-text", "[repetition]") {
    std::vector<std::string> words;
    for (int i = 0; i < 16; ++i) words.push_back("no");
    words.push_back("thanks");
    words.push_back("bye");
    const std::string input = join_words(words);
    const std::string expected = join_words({"no", "no", "no"});
    CHECK(trim_excessive_repetition(input) == expected);
}

TEST_CASE("trim_excessive_repetition trims a repeated two-word phrase at the tail", "[repetition]") {
    std::vector<std::string> words = {"intro", "line"};
    for (int i = 0; i < 4; ++i) {
        words.push_back("go");
        words.push_back("now");
    }
    const std::string input = join_words(words);
    const std::string expected = join_words({"intro", "line", "go", "now", "go", "now"});
    CHECK(trim_excessive_repetition(input) == expected);
}
