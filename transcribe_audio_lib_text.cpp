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
#include "transcribe_audio_lib_text.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <vector>

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
