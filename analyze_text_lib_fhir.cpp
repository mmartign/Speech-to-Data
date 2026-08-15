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
#include "analyze_text_lib_fhir.h"

#include "analyze_text_lib_cleanup.h"

bool is_fhir_bundle_object(const json& candidate) {
    if (!candidate.is_object()) {
        return false;
    }
    const auto type_it = candidate.find("resourceType");
    if (type_it == candidate.end() || !type_it->is_string()) {
        return false;
    }
    return type_it->get<std::string>() == "Bundle";
}

bool extract_fhir_bundle_from_text(const std::string& text, json& bundle, size_t& start_pos, size_t& end_pos) {
    bool in_string = false;
    bool escaped = false;

    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];

        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }

        if (c != '{') {
            continue;
        }

        size_t depth = 0;
        bool local_in_string = false;
        bool local_escaped = false;
        bool completed = false;

        for (size_t j = i; j < text.size(); ++j) {
            const char cj = text[j];
            if (local_in_string) {
                if (local_escaped) {
                    local_escaped = false;
                } else if (cj == '\\') {
                    local_escaped = true;
                } else if (cj == '"') {
                    local_in_string = false;
                }
                continue;
            }

            if (cj == '"') {
                local_in_string = true;
                continue;
            }
            if (cj == '{') {
                ++depth;
            } else if (cj == '}') {
                if (depth == 0) {
                    break;
                }
                --depth;
                if (depth == 0) {
                    const std::string candidate = text.substr(i, j - i + 1);
                    try {
                        json parsed;
                        try {
                            parsed = json::parse(candidate);
                        } catch (...) {
                            parsed = json::parse(strip_json_like_comments(candidate));
                        }
                        if (is_fhir_bundle_object(parsed)) {
                            // Return the first valid Bundle object found in the response text.
                            bundle = std::move(parsed);
                            start_pos = i;
                            end_pos = j + 1;
                            return true;
                        }
                    } catch (...) {
                        // Keep searching for the next candidate JSON object.
                    }
                    completed = true;
                    break;
                }
            }
        }

        if (!completed) {
            break;
        }
    }

    return false;
}

bool extract_revised_bundle(const json& mapper_output, json& revised_bundle) {
    // Accept either raw Bundle output or wrapped mapper payload.
    if (is_fhir_bundle_object(mapper_output)) {
        revised_bundle = mapper_output;
        return true;
    }

    const auto accepted_it = mapper_output.find("acceptedBundle");
    if (accepted_it != mapper_output.end() && is_fhir_bundle_object(*accepted_it)) {
        revised_bundle = *accepted_it;
        return true;
    }

    return false;
}
