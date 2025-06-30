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
import configparser
from openai import OpenAI

# Load configuration
config = configparser.ConfigParser()
config.read("./config.ini")

# Configuration
OPENWEBUI_URL = config["openai"]["base_url"]
API_KEY = config["openai"]["api_key"]
MODEL_NAME = config["openai"]["model_name"]
KNOWLEDGE_BASE_IDS = [config["analysis"]["knowledge_base_ids"]]
COLLECTION = config["analysis"]["collection"] + "\n"
PROMPT = config["prompts"]["prompt"]
TRIGGER_START = config["triggers"]["start"]
TRIGGER_STOP = config["triggers"]["stop"]

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

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": COLLECTION + PROMPT + text}],
                stream=True,
                extra_body={
                    "knowledge_base_ids": KNOWLEDGE_BASE_IDS,
                    "enable_websearch": True
                }
            )

            full_response = ""

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_response += text_chunk

            my_file.write("\n\nFull response received:\n")
            my_file.write(full_response)

        except Exception as e:
            # Catch and log any unexpected error
            error_msg = f"\n[ERROR] Analysis[{analysis_id}] failed: {str(e)}\n"
            print(error_msg)
            my_file.write(error_msg)

    print(f"Processing of Analysis[{analysis_id}] Finished ------------------->>>")
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
                print(f"Analysis has already been started  ------------------->>>")
            else:
                print(f"Analysis started ------------------->>>")
                collected_text = ""
                collect_text = True
                    
        if contains_substring(line.lower(), TRIGGER_STOP.lower()):
            if not collect_text:
                print(f"No analysis is currently running ------------------->>>")
            else:
                if in_analysis.is_set():
                    print(f"A previous analysis is still being processed ------------------->>>")
                else:
                    print(f"Analysis stopped ------------------->>>")
                    collected_text += line
                    threading.Thread(target=analyze_text, args=(collected_text,)).start()
                    collected_text = ""
                    collect_text = False
        
        if collect_text:
                collected_text += line
            


if __name__ == "__main__":
    main()
