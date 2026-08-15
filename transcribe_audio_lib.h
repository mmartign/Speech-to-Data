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

// Pure logic extracted from transcribe_audio.cpp: no PortAudio, no whisper.cpp,
// no Boost, no filesystem/network I/O. Kept separate so it can be linked into
// a unit-test binary without pulling in any of those heavy dependencies.
//
// The logic is split by functional area into the transcribe_audio_lib_*.h/.cpp
// modules below; this header just aggregates them for convenience of callers
// (transcribe_audio.cpp) that want the whole surface.

#include "transcribe_audio_lib_json.h"
#include "transcribe_audio_lib_repetition.h"
#include "transcribe_audio_lib_text.h"
