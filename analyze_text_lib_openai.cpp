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
#include "analyze_text_lib_openai.h"

#include "analyze_text_lib_text.h"

// Safely extract a textual message content from an OpenAI-style response
std::string extract_message_content(const json& response) {
    // Normalize multiple OpenAI response shapes into plain text.
    const auto choices_it = response.find("choices");
    if (choices_it == response.end() || !choices_it->is_array() || choices_it->empty()) {
        return {};
    }

    const auto& first_choice = (*choices_it)[0];
    if (!first_choice.is_object()) {
        return {};
    }

    const auto message_it = first_choice.find("message");
    if (message_it == first_choice.end() || !message_it->is_object()) {
        return {};
    }

    const auto content_it = message_it->find("content");
    if (content_it == message_it->end()) {
        return {};
    }

    if (content_it->is_string()) {
        return trim_whitespace(content_it->get<std::string>());
    }

    if (content_it->is_array()) {
        std::string combined;
        for (const auto& part : *content_it) {
            std::string part_text;
            if (part.is_string()) {
                part_text = part.get<std::string>();
            } else if (part.is_object()) {
                const auto text_it = part.find("text");
                if (text_it != part.end() && text_it->is_string()) {
                    part_text = text_it->get<std::string>();
                } else {
                    const auto content_it2 = part.find("content");
                    if (content_it2 != part.end() && content_it2->is_string()) {
                        part_text = content_it2->get<std::string>();
                    }
                }
            }
            if (part_text.empty()) {
                continue;
            }
            if (!combined.empty()) {
                combined.push_back('\n');
            }
            combined += part_text;
        }
        return trim_whitespace(combined);
    }

    return {};
}

std::string extract_api_error(const json& response) {
    if (!response.is_object()) {
        return {};
    }

    const auto error_it = response.find("error");
    if (error_it != response.end()) {
        if (error_it->is_string()) {
            return error_it->get<std::string>();
        }
        return error_it->dump();
    }

    const auto detail_it = response.find("detail");
    if (detail_it != response.end()) {
        if (detail_it->is_string()) {
            return detail_it->get<std::string>();
        }
        return detail_it->dump();
    }

    const auto message_it = response.find("message");
    if (message_it != response.end() && message_it->is_string()) {
        return message_it->get<std::string>();
    }

    return {};
}
