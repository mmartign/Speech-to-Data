# -*- coding: utf-8 -*-
#
# This file is part of the Spazio IT Speech-to-Data project.
#
# Copyright (C) 2025 Spazio IT 
# Spazio - IT Soluzioni Informatiche s.a.s.
# via Manzoni 40
# 46051 San Giorgio Bigarello
# https://spazioit.com
#
# This file is based on https://github.com/davabase/whisper_real_time, with few modifications
# added to allow specifying the language and enabling usage within a pipeline chain.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.
#
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Real-time transcription using Whisper and SpeechRecognition.")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Model to use.")
    parser.add_argument("--non_english", action="store_true",
                        help="Don't use the English-specific model variant.")
    parser.add_argument("--energy_threshold", type=int, default=1000,
                        help="Energy level for mic to detect.")
    parser.add_argument("--record_timeout", type=float, default=2,
                        help="Time window for real-time audio chunking.")
    parser.add_argument("--phrase_timeout", type=float, default=3,
                        help="Pause duration before starting a new transcription line.")
    parser.add_argument("--language", default="en", choices=["de", "en", "es", "fr", "he", "it", "se"],
                        help="Language for transcription.")
    parser.add_argument("--pipe", action="store_true",
                        help="Enable pipe mode for continuous streaming.")
    
    if 'linux' in platform:
        parser.add_argument("--default_microphone", type=str, default="pulse",
                            help="Default microphone name. Use 'list' to see options.")

    args = parser.parse_args()

    # Variables
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False  # Keep fixed threshold

    # Microphone setup
    if 'linux' in platform:
        mic_name = args.default_microphone
        if mic_name == 'list':
            print("Available microphone devices:")
            for name in sr.Microphone.list_microphone_names():
                print(f"- {name}")
            return
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load Whisper model
    model_name = args.model
    if model_name != "large" and not args.non_english:
        model_name += ".en"
    audio_model = whisper.load_model(model_name)

    transcription = [""]

    # Calibrate microphone
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """Background thread callback to receive audio chunks."""
        data_queue.put(audio.get_raw_data())

    # Start background listening
    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)
    if not args.pipe:
        print("Model loaded.\n")

    # Main loop
    while True:
        try:
            now = datetime.utcnow()

            if not data_queue.empty():
                phrase_complete = False

                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                # Process audio buffer
                audio_data = b"".join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                audio_model.language = args.language
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result["text"].strip()

                if args.pipe:
                    print(text, flush=True)
                else:
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    os.system("cls" if os.name == "nt" else "clear")
                    for line in transcription:
                        print(line)
                    print("", end="", flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    if not args.pipe:
        print("\n\nTranscription:")
        for line in transcription:
            print(line)


if __name__ == "__main__":
    main()
