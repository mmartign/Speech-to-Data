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

#include "../transcribe_audio_lib.h"

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

// ── Generic text helpers ─────────────────────────────────────────────────────

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

// ── WebSocket control-message JSON helpers ──────────────────────────────────

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
