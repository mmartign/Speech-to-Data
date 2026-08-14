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
#include "transcribe_audio_lib.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <sstream>

// ── Transcript word / repetition-trimming helpers ───────────────────────────

bool transcript_word_byte(unsigned char c) {
    // Treat non-ASCII bytes as word bytes so UTF-8 words such as "c'e" or
    // "autorizzazioni" with accents stay in one comparable token.
    return std::isalnum(c) || c >= 0x80 || c == '\'';
}

std::string normalize_transcript_word(const std::string& word) {
    size_t begin = 0;
    size_t end = word.size();
    while (begin < end && word[begin] == '\'') {
        ++begin;
    }
    while (end > begin && word[end - 1] == '\'') {
        --end;
    }

    std::string normalized;
    normalized.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        const unsigned char c = static_cast<unsigned char>(word[i]);
        normalized.push_back(c < 0x80
                                 ? static_cast<char>(std::tolower(c))
                                 : static_cast<char>(c));
    }
    return normalized;
}

std::vector<TranscriptWordSpan> extract_transcript_words(const std::string& text) {
    std::vector<TranscriptWordSpan> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() &&
               !transcript_word_byte(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        const size_t begin = i;
        while (i < text.size() &&
               transcript_word_byte(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        if (begin == i) {
            continue;
        }

        std::string normalized = normalize_transcript_word(text.substr(begin, i - begin));
        if (!normalized.empty()) {
            words.push_back(TranscriptWordSpan{begin, i, std::move(normalized)});
        }
    }
    return words;
}

std::string trim_ascii_edges_copy(const std::string& text) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto first = std::find_if(text.begin(), text.end(), not_space);
    if (first == text.end()) {
        return {};
    }
    const auto last = std::find_if(text.rbegin(), text.rend(), not_space).base();
    return std::string(first, last);
}

bool repeated_word_sequence_equal(const std::vector<TranscriptWordSpan>& words,
                                   size_t lhs,
                                   size_t rhs,
                                   size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (words[lhs + i].normalized != words[rhs + i].normalized) {
            return false;
        }
    }
    return true;
}

std::optional<size_t> find_repetitive_tail_trim_offset(
    const std::vector<TranscriptWordSpan>& words) {
    if (words.size() < 8) {
        return std::nullopt;
    }

    constexpr size_t kAllowedConsecutiveWordRepeats = 3;
    constexpr size_t kMinConsecutiveWordRepeatsAtTail = 8;
    constexpr size_t kMinConsecutiveWordRepeatsAnywhere = 16;
    for (size_t i = 0; i < words.size();) {
        size_t j = i + 1;
        while (j < words.size() && words[j].normalized == words[i].normalized) {
            ++j;
        }

        const size_t run_length = j - i;
        const bool reaches_tail = j == words.size();
        if (run_length >= kMinConsecutiveWordRepeatsAnywhere ||
            (reaches_tail && run_length >= kMinConsecutiveWordRepeatsAtTail)) {
            return words[i + kAllowedConsecutiveWordRepeats].begin;
        }
        i = j;
    }

    constexpr size_t kAllowedPhraseRepeats = 2;
    constexpr size_t kMinPhraseRepeatsAtTail = 4;
    constexpr size_t kMaxPhraseWords = 5;
    for (size_t phrase_words = 2; phrase_words <= kMaxPhraseWords; ++phrase_words) {
        for (size_t i = 0; i + phrase_words * kMinPhraseRepeatsAtTail <= words.size(); ++i) {
            size_t repeats = 1;
            while (i + (repeats + 1) * phrase_words <= words.size() &&
                   repeated_word_sequence_equal(words,
                                                i,
                                                i + repeats * phrase_words,
                                                phrase_words)) {
                ++repeats;
            }
            if (repeats >= kMinPhraseRepeatsAtTail &&
                i + repeats * phrase_words == words.size()) {
                return words[i + phrase_words * kAllowedPhraseRepeats].begin;
            }
        }
    }

    return std::nullopt;
}

std::string trim_excessive_repetition(const std::string& text) {
    const auto words = extract_transcript_words(text);
    const auto trim_offset = find_repetitive_tail_trim_offset(words);
    if (!trim_offset.has_value()) {
        return text;
    }
    return trim_ascii_edges_copy(text.substr(0, *trim_offset));
}

// ── Generic text helpers ─────────────────────────────────────────────────────

std::string trim(const std::string& str) {
    const size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return {};
    }
    const size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

// Whisper emits these tokens when it detects no speech; suppress them so they
// are never written to the transcript or forwarded to WebSocket clients.
bool is_whisper_noise_token(const std::string& text) {
    // Exact-match bracket/paren pseudo-tokens.
    static const std::vector<std::string> kExactTokens = {
        "[BLANK_AUDIO]", "[ Silence]", "[silence]", "(silence)",
        "[Music]", "[ Music]", "(Music)", "(music)", "[music]",
        "[Applause]", "[ Applause]", "(Applause)",
        "[MUSIC]", "[APPLAUSE]",
    };
    for (const auto& tok : kExactTokens) {
        if (text == tok) {
            return true;
        }
    }

    // Normalize: lowercase + strip trailing punctuation/whitespace, then check
    // against known hallucination phrases that Whisper emits on silence/noise.
    std::string norm = text;
    while (!norm.empty() &&
           (std::ispunct(static_cast<unsigned char>(norm.back())) ||
            std::isspace(static_cast<unsigned char>(norm.back())))) {
        norm.pop_back();
    }
    std::transform(norm.begin(), norm.end(), norm.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    static const std::vector<std::string> kHallucinationPhrases = {
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "thank you very much",
        "thank you so much",
        "thanks for listening",
        "thank you for listening",
        "thanks",
        "you",                  // single-token noise on some models
        "bye",
        "bye bye",
        "goodbye",
        // Large-model hallucinations on silence/background noise
        "i believe in the lord",
        "i believe in god",
        "and the door",
        "the door",
        "subscribe",
        "subtitles by",
        "subtitles by the amara.org community",
        "subtitled by",
        "transcribed by",
        "sottotitoli creati dalla comunit\xc3\xa0 amara.org",
        "sottotitoli creati dalla comunit\xc3\xa0 amara org",
        "sottotitoli creati dalla comunita amara.org",
        "sottotitoli creati dalla comunita amara org",
        "www",
    };
    for (const auto& phrase : kHallucinationPhrases) {
        if (norm == phrase) {
            return true;
        }
    }
    return false;
}

std::string format_datetime(const std::chrono::time_point<std::chrono::system_clock>& tp) {
    const std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "[%Y-%m-%d %H:%M:%S]");
    return oss.str();
}

// ── WebSocket control-message JSON helpers (flat, shallow parser) ──────────

std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

size_t skip_json_whitespace(const std::string& json, size_t pos) {
    while (pos < json.size()) {
        const unsigned char c = static_cast<unsigned char>(json[pos]);
        if (!std::isspace(c)) {
            break;
        }
        ++pos;
    }
    return pos;
}

std::optional<std::string> parse_json_quoted_string(const std::string& json, size_t& pos) {
    if (pos >= json.size() || json[pos] != '"') {
        return std::nullopt;
    }
    ++pos;
    std::string out;
    out.reserve(16);
    while (pos < json.size()) {
        const char c = json[pos++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (pos >= json.size()) {
                return std::nullopt;
            }
            const char esc = json[pos++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                // Keep unicode escapes literal: protocol routing fields are
                // ASCII names and ids, so full Unicode decoding is unnecessary.
                case 'u': out += "\\u"; break;
                default: out.push_back(esc); break;
            }
            continue;
        }
        out.push_back(c);
    }
    return std::nullopt;
}

std::string extract_json_string_field(const std::string& json, const std::string& field) {
    // This parser is intentionally shallow because start/stop frames are flat
    // JSON objects controlled by our mobile client.
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return {};
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return {};
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    auto parsed = parse_json_quoted_string(json, value_pos);
    return parsed.has_value() ? *parsed : std::string{};
}

int extract_json_int_field(const std::string& json, const std::string& field, int fallback) {
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return fallback;
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    size_t end = value_pos;
    if (end < json.size() && (json[end] == '-' || json[end] == '+')) {
        ++end;
    }
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) {
        ++end;
    }
    if (end == value_pos || (end == value_pos + 1 && (json[value_pos] == '-' || json[value_pos] == '+'))) {
        return fallback;
    }
    try {
        return std::stoi(json.substr(value_pos, end - value_pos));
    } catch (...) {
        return fallback;
    }
}

std::string ascii_lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim_ascii_whitespace(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto first = std::find_if(s.begin(), s.end(), not_space);
    if (first == s.end()) {
        return {};
    }
    const auto last = std::find_if(s.rbegin(), s.rend(), not_space).base();
    return std::string(first, last);
}

bool json_value_delimiter(char c) {
    return c == ',' || c == '}' || c == ']' || std::isspace(static_cast<unsigned char>(c));
}

bool extract_json_bool_field(const std::string& json, const std::string& field, bool fallback) {
    // Accept both real booleans and string-ish form values to keep the protocol
    // tolerant of clients that serialize UI settings as strings.
    const std::string key = std::string("\"") + field + "\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = skip_json_whitespace(json, pos + key.size());
    if (value_pos >= json.size() || json[value_pos] != ':') {
        return fallback;
    }
    value_pos = skip_json_whitespace(json, value_pos + 1);
    if (value_pos >= json.size()) {
        return fallback;
    }

    if (json[value_pos] == '"') {
        auto parsed = parse_json_quoted_string(json, value_pos);
        if (!parsed.has_value()) {
            return fallback;
        }
        const std::string value = ascii_lower_copy(trim_ascii_whitespace(*parsed));
        if (value == "true" || value == "1" || value == "on" || value == "yes") {
            return true;
        }
        if (value == "false" || value == "0" || value == "off" || value == "no") {
            return false;
        }
        return fallback;
    }

    const size_t remaining = json.size() - value_pos;
    if (remaining >= 4 &&
        json.compare(value_pos, 4, "true") == 0 &&
        (value_pos + 4 == json.size() || json_value_delimiter(json[value_pos + 4]))) {
        return true;
    }
    if (remaining >= 5 &&
        json.compare(value_pos, 5, "false") == 0 &&
        (value_pos + 5 == json.size() || json_value_delimiter(json[value_pos + 5]))) {
        return false;
    }
    if (json[value_pos] == '1' &&
        (value_pos + 1 == json.size() || json_value_delimiter(json[value_pos + 1]))) {
        return true;
    }
    if (json[value_pos] == '0' &&
        (value_pos + 1 == json.size() || json_value_delimiter(json[value_pos + 1]))) {
        return false;
    }
    return fallback;
}

bool json_message_type_is(const std::string& json, const char* expected) {
    return extract_json_string_field(json, "type") == expected;
}
