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
#include "transcribe_audio_lib_repetition.h"

#include <algorithm>
#include <cctype>

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
