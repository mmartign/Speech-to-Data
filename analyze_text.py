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
import sys
import threading
from openai import OpenAI

# Configuration
OPENWEBUI_URL = "http://localhost:8080/api"
API_KEY = "my-key"
MODEL_NAME = "deepseek-r1:32b"
KNOWLEDGE_BASE_IDS = ["#Treatment_Protocols"]  # The treatment protocols actually available
COLLECTION = "#Treatment_Protocols\n"
PROMPT = (
    "Please, begin by extracting the medical data from the following text and converting it into FHIR compliant JSON resource bundle.\n"
    "After that, verify whether the operators have adhered to any of the attached protocols.\n\n"
)
PROMPT_NON_ENGLISH = (
    "Please, begin by translating the following Italian text into English.\n"
    "After that,  extract the medical data from the translated text and convert it into FHIR compliant JSON resource bundle..\n"
    "Finally, verify whether the operators have adhered to any of the attached protocols.\n\n"
)
TRIGGER_START = "Start analysis"
TRIGGER_STOP = "Stop analysis"

# Global state
analysis_counter = threading.Lock()
counter_value = 0
in_analysis = threading.Event()


def analysis_num():
    """Thread-safe increment of analysis counter."""
    global counter_value
    with analysis_counter:
        counter_value += 1
        return counter_value


def contains_substring(string, substring):
    """Return True if substring is in string or if substring is empty."""
    return not substring or substring in string


def analyze_text(text):
    """Perform AI-based analysis on input text in a separate thread."""
    in_analysis.set()
    analysis_id = analysis_num()
    print(f"Processing of Analysis[{analysis_id}] Started ------------------->>>")

    client = OpenAI(base_url=OPENWEBUI_URL, api_key=API_KEY)
    filename = f"results_analysis{analysis_id}.txt"

    with open(filename, "w", encoding="utf-8") as my_file:
        my_file.write(f"\nUsing model: {MODEL_NAME}\n")
        my_file.write(f"Prompt: {COLLECTION + PROMPT + text}\n")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": COLLECTION + PROMPT + text}],
            stream=True,
            extra_body={
                "knowledge_base_ids": KNOWLEDGE_BASE_IDS
            }
        )

        full_response = ""

        for chunk in response:
            if chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                full_response += text_chunk

        my_file.write("\n\nFull response received:\n")
        my_file.write(full_response)

    print(f"Processing of Analysis[{analysis_id}] Finished <<<-------------------")
    in_analysis.clear()


def main():
    """Main loop listening to stdin for input and triggers."""
    print("Listening for input...")
    
    collected_text = ""
    collect_text = False

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        print(line.strip())
        
        if contains_substring(line.lower(), TRIGGER_START.lower()):
            if collect_text:
                print(f"Aanalysis has already been started.")
            else:
                collected_text = ""
                collect_text = True
                    
        if contains_substring(line.lower(), TRIGGER_STOP.lower()):
            if not collect_text:
                print(f"No analysis is currently running.")
            else:
                if in_analysis.is_set():
                    print(f"A previous analysis is still being processed")
                else:
                    collected_text += line
                    threading.Thread(target=analyze_text, args=(collected_text,)).start()
                    collected_text = ""
                    collect_text = False
        
        if collect_text:
                collected_text += line
            


if __name__ == "__main__":
    main()
