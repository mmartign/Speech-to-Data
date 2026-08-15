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

// Transcript word / repetition-trimming helpers extracted from
// transcribe_audio.cpp.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

struct TranscriptWordSpan {
    size_t begin = 0;
    size_t end = 0;
    std::string normalized;
};

bool transcript_word_byte(unsigned char c);
std::string normalize_transcript_word(const std::string& word);
std::vector<TranscriptWordSpan> extract_transcript_words(const std::string& text);
std::string trim_ascii_edges_copy(const std::string& text);
bool repeated_word_sequence_equal(const std::vector<TranscriptWordSpan>& words,
                                   size_t lhs, size_t rhs, size_t count);
std::optional<size_t> find_repetitive_tail_trim_offset(const std::vector<TranscriptWordSpan>& words);
std::string trim_excessive_repetition(const std::string& text);
