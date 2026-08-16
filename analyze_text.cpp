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
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <cctype>
#include <chrono>
#include <openai.hpp>

#include "analyze_text_lib.h"

using json = nlohmann::json;

// Global runtime configuration loaded from config.ini.
std::string OPENWEBUI_URL;
std::string API_KEY;
std::string MODEL_NAME;
std::string KNOWLEDGE_BASE_IDS;
std::string PROMPT;
std::string TEMP_PROMPT;
std::string HELP_PROMPT;
std::string TRIGGER_START;
std::string TRIGGER_STOP;
std::string TRIGGER_TEMP_CHECK;
std::string TRIGGER_HELP;
std::string TRIGGER_DISCARD;
std::string TRIGGER_REPEAT;
std::string TRIGGER_STATUS;
std::string TRIGGER_PAUSE;
std::string TRIGGER_RESUME;
std::string TRIGGER_LIST_COMMANDS;
std::string TRIGGER_CAMERA_ON;
std::string TRIGGER_CAMERA_OFF;
std::string TTS_COMMAND;
double SELF_ECHO_GRACE_SECONDS = 5.0;
bool MAPPER_NETWORK_ENABLED = true;
std::string MAPPER_CACHE_DIR = "./terminology_cache";
std::string MAPPER_CACHE_TTL_DAYS = "7";
std::string MAPPER_LOINC_USER;
std::string MAPPER_LOINC_PASS;
std::string MAPPER_TIMEOUT_SECONDS = "10";
std::string CAMERA_ID = "0";
std::string CAMERA_INTERVAL = "10";

std::mutex analysis_mutex;
std::atomic<int> counter_value{0};
std::atomic<int> temp_counter_value{0};
std::atomic<int> help_counter_value{0};
std::atomic<int> camera_counter_value{0};
std::atomic<int> active_analyses{0};
std::mutex tts_mutex;
std::mutex feedback_mutex;
std::string last_feedback_message;
std::atomic<int64_t> self_echo_mute_until_ms{0};
std::once_flag openai_init_flag;
bool check_fhir = false;
bool no_analysis_summary = false;

// ── Language / i18n ──────────────────────────────────────────────────────────

enum class Lang { EN, IT, FR };
static Lang g_lang = Lang::EN;

enum MsgKey {
    MSG_UNABLE_TO_OPEN_CONFIG = 0,
    MSG_MISSING_REQUIRED_CONFIG,
    MSG_KB_NOT_SET,
    MSG_ERR_OPEN_RESULTS,
    MSG_WARN_NO_TEXT_PREFIX,
    MSG_WARN_NO_TEXT_SUFFIX,
    MSG_ERR_ANALYSIS_FAILED_PREFIX,
    MSG_ERR_ANALYSIS_FAILED_MIDDLE,
    MSG_WARN_SUMMARY_NO_TEXT_PREFIX,
    MSG_WARN_SUMMARY_NO_TEXT_SUFFIX,
    MSG_ERR_SUMMARY_FAILED_PREFIX,
    MSG_ERR_SUMMARY_FAILED_MIDDLE,
    MSG_ERR_WRITE_RESULTS_PREFIX,
    MSG_ERR_WRITE_RESULTS_SUFFIX,
    MSG_ERR_WRITE_HEADER,
    MSG_FAILED_LOAD_CONFIG,
    MSG_LISTENING,
    MSG_ANALYSIS_STARTED_PREFIX,
    MSG_ANALYSIS_STARTED_SUFFIX,
    MSG_ANALYSIS_FINISHED_SUFFIX,
    MSG_TEMP_ANALYSIS_STARTED_PREFIX,
    MSG_TEMP_ANALYSIS_FINISHED_PREFIX,
    MSG_RECORDING_ALREADY_STARTED,
    MSG_RECORDING_STARTED,
    MSG_NO_RECORDING_RUNNING,
    MSG_RECORDING_STOPPED,
    MSG_ANOTHER_ANALYSIS_RUNNING,
    MSG_ANALYSIS_RUNNING_STOP_BLOCKED,
    MSG_ANALYSIS_FEEDBACK_PREFIX,
    MSG_ANALYSIS_SUMMARY_FEEDBACK_MIDDLE,
    MSG_TEMP_CHECK_REQUESTED,
    MSG_HELP_ANALYSIS_STARTED_PREFIX,
    MSG_HELP_ANALYSIS_FINISHED_PREFIX,
    MSG_HELP_REQUESTED,
    MSG_RECORDING_DISCARDED,
    MSG_NOTHING_TO_REPEAT,
    MSG_STATUS_HEADER,
    MSG_STATUS_STATE_IDLE,
    MSG_STATUS_STATE_COLLECTING,
    MSG_STATUS_STATE_PAUSED,
    MSG_STATUS_ANALYSES_MIDDLE,
    MSG_RECORDING_PAUSED,
    MSG_RECORDING_ALREADY_PAUSED,
    MSG_RECORDING_RESUMED,
    MSG_RECORDING_NOT_PAUSED,
    MSG_LIST_COMMANDS_HEADER,
    MSG_CAMERA_ON_REQUESTED,
    MSG_CAMERA_OFF_REQUESTED,
    MSG_STATUS_CAMERA_MIDDLE,
    MSG_STATUS_CAMERA_ON,
    MSG_STATUS_CAMERA_OFF,
    MSG_COUNT
};

enum class RecordingState { Idle, Collecting, Paused };

// Columns: EN=0, IT=1, FR=2
static const char* MESSAGES[MSG_COUNT][3] = {
    /* MSG_UNABLE_TO_OPEN_CONFIG */
    {"Unable to open config file: ",
     "Impossibile aprire il file di configurazione: ",
     "Impossible d'ouvrir le fichier de configuration : "},
    /* MSG_MISSING_REQUIRED_CONFIG */
    {"Missing required config values:",
     "Valori di configurazione richiesti mancanti:",
     "Valeurs de configuration requises manquantes :"},
    /* MSG_KB_NOT_SET */
    {"Warning: analysis.knowledge_base_ids is not set; knowledge base lookups will be skipped.\n",
     "Attenzione: analysis.knowledge_base_ids non è impostato; le ricerche nella knowledge base verranno saltate.\n",
     "Avertissement : analysis.knowledge_base_ids n'est pas défini ; les recherches dans la base de connaissances seront ignorées.\n"},
    /* MSG_ERR_OPEN_RESULTS */
    {"[ERROR] Unable to open results file: ",
     "[ERRORE] Impossibile aprire il file dei risultati: ",
     "[ERREUR] Impossible d'ouvrir le fichier de résultats : "},
    /* MSG_WARN_NO_TEXT_PREFIX */
    {"[WARN] Analysis[",
     "[AVVISO] Analisi[",
     "[AVERT] Analyse["},
    /* MSG_WARN_NO_TEXT_SUFFIX */
    {"] returned no text content; see results file.\n",
     "] non ha restituito contenuto testuale; vedere il file dei risultati.\n",
     "] n'a retourné aucun contenu textuel ; voir le fichier de résultats.\n"},
    /* MSG_ERR_ANALYSIS_FAILED_PREFIX */
    {"[ERROR] Analysis[",
     "[ERRORE] Analisi[",
     "[ERREUR] Analyse["},
    /* MSG_ERR_ANALYSIS_FAILED_MIDDLE */
    {"] failed: ",
     "] fallita: ",
     "] échouée : "},
    /* MSG_WARN_SUMMARY_NO_TEXT_PREFIX */
    {"[WARN] Summary generation returned no text for Analysis[",
     "[AVVISO] La generazione del riepilogo non ha restituito testo per Analisi[",
     "[AVERT] La génération du résumé n'a retourné aucun texte pour Analyse["},
    /* MSG_WARN_SUMMARY_NO_TEXT_SUFFIX */
    {"]; see results file.\n",
     "]; vedere il file dei risultati.\n",
     "] ; voir le fichier de résultats.\n"},
    /* MSG_ERR_SUMMARY_FAILED_PREFIX */
    {"[ERROR] Summary generation failed for Analysis[",
     "[ERRORE] Generazione del riepilogo fallita per Analisi[",
     "[ERREUR] Génération du résumé échouée pour Analyse["},
    /* MSG_ERR_SUMMARY_FAILED_MIDDLE */
    {"]: ",
     "]: ",
     "] : "},
    /* MSG_ERR_WRITE_RESULTS_PREFIX */
    {"[ERROR] Writing to results file failed for Analysis[",
     "[ERRORE] Scrittura nel file dei risultati fallita per Analisi[",
     "[ERREUR] Échec d'écriture dans le fichier de résultats pour Analyse["},
    /* MSG_ERR_WRITE_RESULTS_SUFFIX */
    {"]\n", "]\n", "]\n"},
    /* MSG_ERR_WRITE_HEADER */
    {"[ERROR] Failed to write analysis header to ",
     "[ERRORE] Impossibile scrivere l'intestazione dell'analisi in ",
     "[ERREUR] Impossible d'écrire l'en-tête d'analyse dans "},
    /* MSG_FAILED_LOAD_CONFIG */
    {"Failed to load config.ini\n",
     "Impossibile caricare config.ini\n",
     "Impossible de charger config.ini\n"},
    /* MSG_LISTENING */
    {"Listening for input...\n",
     "In ascolto per l'input...\n",
     "En attente d'entrée...\n"},
    /* MSG_ANALYSIS_STARTED_PREFIX */
    {"Analysis of Recording[",
     "Analisi della Registrazione[",
     "Analyse de l'Enregistrement["},
    /* MSG_ANALYSIS_STARTED_SUFFIX */
    {"] Started ------------------->>>\n",
     "] Avviata ------------------->>>\n",
     "] Démarrée ------------------->>>\n"},
    /* MSG_ANALYSIS_FINISHED_SUFFIX */
    {"] Finished ------------------->>>\n",
     "] Completata ------------------->>>\n",
     "] Terminée ------------------->>>\n"},
    /* MSG_TEMP_ANALYSIS_STARTED_PREFIX */
    {"Temporary_Analysis of Recording[",
     "Analisi_Temporanea della Registrazione[",
     "Analyse_Temporaire de l'Enregistrement["},
    /* MSG_TEMP_ANALYSIS_FINISHED_PREFIX */
    {"Temporary Analysis of Recording[",
     "Analisi Temporanea della Registrazione[",
     "Analyse Temporaire de l'Enregistrement["},
    /* MSG_RECORDING_ALREADY_STARTED */
    {"Recording has already been started ------------------->>>\n",
     "La registrazione è già stata avviata ------------------->>>\n",
     "L'enregistrement a déjà été démarré ------------------->>>\n"},
    /* MSG_RECORDING_STARTED */
    {"Recording started ------------------->>>\n",
     "Registrazione avviata ------------------->>>\n",
     "Enregistrement démarré ------------------->>>\n"},
    /* MSG_NO_RECORDING_RUNNING */
    {"No recording is currently running ------------------->>>\n",
     "Nessuna registrazione è in corso ------------------->>>\n",
     "Aucun enregistrement n'est en cours ------------------->>>\n"},
    /* MSG_RECORDING_STOPPED */
    {"Recording stopped ------------------->>>\n",
     "Registrazione fermata ------------------->>>\n",
     "Enregistrement arrêté ------------------->>>\n"},
    /* MSG_ANOTHER_ANALYSIS_RUNNING */
    {"Another analysis is running; this one will start once it finishes ------------------->>>\n",
     "Un'altra analisi è in corso; questa inizierà al termine ------------------->>>\n",
     "Une autre analyse est en cours ; celle-ci démarrera une fois terminée ------------------->>>\n"},
    /* MSG_ANALYSIS_RUNNING_STOP_BLOCKED */
    {"An analysis is still running; recording continues, stop again once it finishes ------------------->>>\n",
     "Un'analisi è ancora in corso; la registrazione continua, fermala di nuovo al termine ------------------->>>\n",
     "Une analyse est encore en cours ; l'enregistrement continue, arrêtez-le à nouveau une fois terminée ------------------->>>\n"},
    /* MSG_ANALYSIS_FEEDBACK_PREFIX */
    {"Analysis[",
     "Analisi[",
     "Analyse["},
    /* MSG_ANALYSIS_SUMMARY_FEEDBACK_MIDDLE */
    {"] completed. Summary: ",
     "] completata. Riepilogo: ",
     "] terminée. Résumé : "},
    /* MSG_TEMP_CHECK_REQUESTED */
    {"Temporary check requested ------------------->>>\n",
     "Controllo temporaneo richiesto ------------------->>>\n",
     "Vérification temporaire demandée ------------------->>>\n"},
    /* MSG_HELP_ANALYSIS_STARTED_PREFIX */
    {"Help_Analysis of Recording[",
     "Analisi_di_Aiuto della Registrazione[",
     "Analyse_d'Aide de l'Enregistrement["},
    /* MSG_HELP_ANALYSIS_FINISHED_PREFIX */
    {"Help Analysis of Recording[",
     "Analisi di Aiuto della Registrazione[",
     "Analyse d'Aide de l'Enregistrement["},
    /* MSG_HELP_REQUESTED */
    {"Help requested ------------------->>>\n",
     "Aiuto richiesto ------------------->>>\n",
     "Aide demandée ------------------->>>\n"},
    /* MSG_RECORDING_DISCARDED */
    {"Recording discarded ------------------->>>\n",
     "Registrazione scartata ------------------->>>\n",
     "Enregistrement annulé ------------------->>>\n"},
    /* MSG_NOTHING_TO_REPEAT */
    {"Nothing to repeat yet ------------------->>>\n",
     "Niente da ripetere ancora ------------------->>>\n",
     "Rien à répéter pour le moment ------------------->>>\n"},
    /* MSG_STATUS_HEADER */
    {"Status: recording is ",
     "Stato: la registrazione è ",
     "État : l'enregistrement est "},
    /* MSG_STATUS_STATE_IDLE */
    {"off", "spenta", "arrêté"},
    /* MSG_STATUS_STATE_COLLECTING */
    {"on", "attiva", "en cours"},
    /* MSG_STATUS_STATE_PAUSED */
    {"paused", "in pausa", "en pause"},
    /* MSG_STATUS_ANALYSES_MIDDLE */
    {". Active analyses: ",
     ". Analisi attive: ",
     ". Analyses actives : "},
    /* MSG_RECORDING_PAUSED */
    {"Recording paused ------------------->>>\n",
     "Registrazione in pausa ------------------->>>\n",
     "Enregistrement mis en pause ------------------->>>\n"},
    /* MSG_RECORDING_ALREADY_PAUSED */
    {"Recording is already paused ------------------->>>\n",
     "La registrazione è già in pausa ------------------->>>\n",
     "L'enregistrement est déjà en pause ------------------->>>\n"},
    /* MSG_RECORDING_RESUMED */
    {"Recording resumed ------------------->>>\n",
     "Registrazione ripresa ------------------->>>\n",
     "Enregistrement repris ------------------->>>\n"},
    /* MSG_RECORDING_NOT_PAUSED */
    {"Recording is not paused ------------------->>>\n",
     "La registrazione non è in pausa ------------------->>>\n",
     "L'enregistrement n'est pas en pause ------------------->>>\n"},
    /* MSG_LIST_COMMANDS_HEADER */
    {"Available voice commands: ",
     "Comandi vocali disponibili: ",
     "Commandes vocales disponibles : "},
    /* MSG_CAMERA_ON_REQUESTED */
    {"Camera on requested ------------------->>>\n",
     "Accensione camera richiesta ------------------->>>\n",
     "Activation de la caméra demandée ------------------->>>\n"},
    /* MSG_CAMERA_OFF_REQUESTED */
    {"Camera off requested ------------------->>>\n",
     "Spegnimento camera richiesto ------------------->>>\n",
     "Désactivation de la caméra demandée ------------------->>>\n"},
    /* MSG_STATUS_CAMERA_MIDDLE */
    {". Camera is ",
     ". Camera: ",
     ". Caméra : "},
    /* MSG_STATUS_CAMERA_ON */
    {"on", "accesa", "allumée"},
    /* MSG_STATUS_CAMERA_OFF */
    {"off", "spenta", "éteinte"},
};

static const char* tr(MsgKey key) {
    const int idx = (g_lang == Lang::IT) ? 1 : (g_lang == Lang::FR) ? 2 : 0;
    return MESSAGES[key][idx];
}

static const char* summary_prompt_for_language() {
    switch (g_lang) {
        case Lang::IT:
            return "Fornisci solo un riepilogo conciso (massimo 3 frasi brevi) in italiano del seguente testo. Non includere ragionamenti interni, tag o passaggi di analisi.\n";
        case Lang::FR:
            return "Fournis uniquement un résumé concis (maximum 3 phrases courtes) en français du texte suivant. N'inclus pas de raisonnement interne, de balises ni d'étapes d'analyse.\n";
        case Lang::EN:
        default:
            return "Provide only a concise summary (max 3 short sentences) in English of the following text. Do not include internal reasoning, tags, or analysis steps.\n";
    }
}

// ── End i18n ─────────────────────────────────────────────────────────────────

class AnalysisSession {
public:
    explicit AnalysisSession(std::mutex& mutex)
        : lock_(mutex) {
        // Serialize analysis sections that share output/logging resources.
    }

private:
    std::unique_lock<std::mutex> lock_;
};

class AnalysisJobGuard {
public:
    explicit AnalysisJobGuard(std::atomic<int>& active_count)
        : active_count_(active_count) {}

    ~AnalysisJobGuard() {
        --active_count_;
    }

private:
    std::atomic<int>& active_count_;
};

std::string revise_fhir_bundle_in_response(std::string response_text,
                                           const std::string& analysis_label,
                                           std::ofstream& file) {
    json detected_bundle;
    size_t start_pos = 0;
    size_t end_pos = 0;
    if (!extract_fhir_bundle_from_text(response_text, detected_bundle, start_pos, end_pos)) {
        // Fast path: no Bundle detected, return original model output.
        return response_text;
    }

    const std::string input_path = "tmp_mapper_input_" + analysis_label + ".json";
    const std::string output_path = "tmp_mapper_output_" + analysis_label + ".json";

    try {
        std::ofstream input_file(input_path);
        if (!input_file.is_open()) {
            file << "\n[WARN] Detected FHIR Bundle but failed to open mapper input file: "
                 << input_path << "\n";
            return response_text;
        }
        input_file << detected_bundle.dump(2) << "\n";
    } catch (const std::exception& e) {
        file << "\n[WARN] Failed to write mapper input bundle: " << e.what() << "\n";
        return response_text;
    }

    std::ostringstream cmd_builder;
    // Shell out to deterministic mapper to post-process model-generated FHIR.
    cmd_builder << "./deterministic_fhir_mapper.exe '" << escape_for_single_quotes(input_path)
                << "' '" << escape_for_single_quotes(output_path)
                << "' --model-name '" << escape_for_single_quotes(MODEL_NAME) << "'";
    if (!MAPPER_NETWORK_ENABLED) {
        cmd_builder << " --no-network";
    }
    if (!MAPPER_CACHE_DIR.empty()) {
        cmd_builder << " --cache-dir '" << escape_for_single_quotes(MAPPER_CACHE_DIR) << "'";
    }
    if (!MAPPER_CACHE_TTL_DAYS.empty()) {
        cmd_builder << " --cache-ttl-days '" << escape_for_single_quotes(MAPPER_CACHE_TTL_DAYS) << "'";
    }
    if (!MAPPER_LOINC_USER.empty()) {
        cmd_builder << " --loinc-user '" << escape_for_single_quotes(MAPPER_LOINC_USER) << "'";
    }
    if (!MAPPER_LOINC_PASS.empty()) {
        cmd_builder << " --loinc-pass '" << escape_for_single_quotes(MAPPER_LOINC_PASS) << "'";
    }
    if (!MAPPER_TIMEOUT_SECONDS.empty()) {
        cmd_builder << " --timeout '" << escape_for_single_quotes(MAPPER_TIMEOUT_SECONDS) << "'";
    }
    cmd_builder << " >/dev/null 2>&1";
    const std::string cmd = cmd_builder.str();
    const int mapper_rc = std::system(cmd.c_str());
    if (mapper_rc != 0) {
        file << "\n[WARN] deterministic_fhir_mapper returned non-zero status (" << mapper_rc
             << "). Keeping original bundle.\n";
        std::remove(input_path.c_str());
        std::remove(output_path.c_str());
        return response_text;
    }

    try {
        std::ifstream output_file(output_path);
        if (!output_file.is_open()) {
            file << "\n[WARN] Mapper output file not found: " << output_path
                 << ". Keeping original bundle.\n";
            std::remove(input_path.c_str());
            std::remove(output_path.c_str());
            return response_text;
        }

        json mapper_output;
        output_file >> mapper_output;

        json revised_bundle;
        if (!extract_revised_bundle(mapper_output, revised_bundle)) {
            file << "\n[WARN] Mapper output did not contain a revised Bundle. Keeping original bundle.\n";
            std::remove(input_path.c_str());
            std::remove(output_path.c_str());
            return response_text;
        }

        const std::string revised_text = revised_bundle.dump(2);
        // Replace only the detected Bundle span, leaving surrounding narrative intact.
        response_text.replace(start_pos, end_pos - start_pos, revised_text);
        file << "\n[INFO] FHIR Bundle detected and revised by deterministic_fhir_mapper.\n";
    } catch (const std::exception& e) {
        file << "\n[WARN] Failed to parse mapper output: " << e.what()
             << ". Keeping original bundle.\n";
    }

    std::remove(input_path.c_str());
    std::remove(output_path.c_str());
    return response_text;
}

void speak_text(const std::string& text, bool wait_for_completion = false) {
    std::string trimmed = strip_trailing_newlines(text);
    if (trimmed.empty()) {
        return;
    }

    trimmed = "SI-Listener Assistant: " + trimmed;

    const std::string escaped = escape_for_single_quotes(trimmed);
    std::string cmd = TTS_COMMAND + " '" + escaped + "' >/dev/null 2>&1";
    if (!wait_for_completion) {
        cmd += " &";
    }

    // TTS backend is shared; avoid overlapping command writes.
    std::lock_guard<std::mutex> lock(tts_mutex);
    std::system(cmd.c_str());
}

void speak_feedback(const std::string& text) {
    // Remember substantive spoken feedback (summaries/suggestions) so it can be replayed on request.
    {
        std::lock_guard<std::mutex> lock(feedback_mutex);
        last_feedback_message = text;
    }
    speak_text(text);
}

// After the mic could plausibly have heard the assistant speak (e.g. the
// list_commands announcement, which recites every trigger phrase verbatim),
// suppress trigger detection for a grace period. This is on top of blocking
// through the TTS call itself; the grace period absorbs the extra latency of
// the transcribe_audio -> ASR pipeline (a separate process that keeps
// listening throughout, independent of our blocking) still emitting a
// trailing transcript line for audio it captured just before playback ended.
// Whisper-based transcription of a multi-second announcement can itself take
// several seconds, so the default is deliberately generous; tune it via
// tts.self_echo_grace_seconds in config.ini for the ASR model/hardware in use.
void mute_self_echo_for(std::chrono::milliseconds duration) {
    const auto mute_until = std::chrono::steady_clock::now() + duration;
    const int64_t mute_until_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        mute_until.time_since_epoch()).count();

    // Extend the mute window rather than shortening an already-longer one.
    int64_t previous = self_echo_mute_until_ms.load();
    while (mute_until_ms > previous &&
           !self_echo_mute_until_ms.compare_exchange_weak(previous, mute_until_ms)) {}
}

bool is_self_echo_muted() {
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return now_ms < self_echo_mute_until_ms.load();
}

void say_info(const std::string& message) {
    std::cout << message;
    speak_text(message);
}

void say_error(const std::string& message) {
    std::cerr << message;
    speak_text(message);
}

// Load config
bool load_config(const std::string& path) {
    std::ifstream file_check(path);
    if (!file_check.is_open()) {
        say_error(tr(MSG_UNABLE_TO_OPEN_CONFIG) + path + "\n");
        return false;
    }
    file_check.close();

    auto config = parse_ini(path);

    std::vector<std::string> missing_keys;
    auto require_value = [&](const std::string& key, std::string& destination) {
        auto it = config.find(key);
        if (it == config.end() || it->second.empty()) {
            missing_keys.push_back(key);
            return;
        }
        destination = it->second;
    };

    require_value("openai.base_url", OPENWEBUI_URL);
    require_value("openai.api_key", API_KEY);
    require_value("openai.model_name", MODEL_NAME);
    require_value("prompts.prompt", PROMPT);
    require_value("prompts.temp_prompt", TEMP_PROMPT);
    require_value("prompts.help_prompt", HELP_PROMPT);
    require_value("triggers.start", TRIGGER_START);
    require_value("triggers.stop", TRIGGER_STOP);
    require_value("triggers.temp_check", TRIGGER_TEMP_CHECK);
    require_value("triggers.help", TRIGGER_HELP);
    require_value("triggers.discard", TRIGGER_DISCARD);
    require_value("triggers.repeat", TRIGGER_REPEAT);
    require_value("triggers.status", TRIGGER_STATUS);
    require_value("triggers.pause", TRIGGER_PAUSE);
    require_value("triggers.resume", TRIGGER_RESUME);
    require_value("triggers.list_commands", TRIGGER_LIST_COMMANDS);
    require_value("triggers.camera_on", TRIGGER_CAMERA_ON);
    require_value("triggers.camera_off", TRIGGER_CAMERA_OFF);
    require_value("tts.command", TTS_COMMAND);

    auto kb_it = config.find("analysis.knowledge_base_ids");
    // Optional: empty means no KB augmentation, not a hard failure.
    KNOWLEDGE_BASE_IDS = (kb_it != config.end()) ? kb_it->second : std::string{};

    auto mapper_network_it = config.find("deterministic_mapper.network_enabled");
    if (mapper_network_it != config.end()) {
        std::string value = mapper_network_it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        MAPPER_NETWORK_ENABLED = !(value == "false" || value == "0" || value == "no" || value == "off");
    }

    auto mapper_cache_dir_it = config.find("deterministic_mapper.cache_dir");
    if (mapper_cache_dir_it != config.end() && !mapper_cache_dir_it->second.empty()) {
        MAPPER_CACHE_DIR = mapper_cache_dir_it->second;
    }

    auto mapper_cache_ttl_it = config.find("deterministic_mapper.cache_ttl_days");
    if (mapper_cache_ttl_it != config.end() && !mapper_cache_ttl_it->second.empty()) {
        MAPPER_CACHE_TTL_DAYS = mapper_cache_ttl_it->second;
    }

    auto mapper_loinc_user_it = config.find("deterministic_mapper.loinc_user");
    if (mapper_loinc_user_it != config.end()) {
        MAPPER_LOINC_USER = mapper_loinc_user_it->second;
    }

    auto mapper_loinc_pass_it = config.find("deterministic_mapper.loinc_pass");
    if (mapper_loinc_pass_it != config.end()) {
        MAPPER_LOINC_PASS = mapper_loinc_pass_it->second;
    }

    auto mapper_timeout_it = config.find("deterministic_mapper.timeout_seconds");
    if (mapper_timeout_it != config.end() && !mapper_timeout_it->second.empty()) {
        MAPPER_TIMEOUT_SECONDS = mapper_timeout_it->second;
    }

    auto self_echo_grace_it = config.find("tts.self_echo_grace_seconds");
    if (self_echo_grace_it != config.end() && !self_echo_grace_it->second.empty()) {
        try {
            SELF_ECHO_GRACE_SECONDS = std::stod(self_echo_grace_it->second);
        } catch (const std::exception&) {
            // Keep the default on an invalid value.
        }
    }

    auto camera_id_it = config.find("camera.camera_id");
    if (camera_id_it != config.end() && !camera_id_it->second.empty()) {
        CAMERA_ID = camera_id_it->second;
    }

    auto camera_interval_it = config.find("camera.camera_interval");
    if (camera_interval_it != config.end() && !camera_interval_it->second.empty()) {
        CAMERA_INTERVAL = camera_interval_it->second;
    }

    if (!missing_keys.empty()) {
        std::ostringstream oss;
        oss << tr(MSG_MISSING_REQUIRED_CONFIG);
        for (const auto& key : missing_keys) {
            oss << ' ' << key;
        }
        oss << "\n";
        say_error(oss.str());
        return false;
    }

    std::transform(TRIGGER_START.begin(), TRIGGER_START.end(), TRIGGER_START.begin(), ::tolower);
    std::transform(TRIGGER_STOP.begin(), TRIGGER_STOP.end(), TRIGGER_STOP.begin(), ::tolower);
    std::transform(TRIGGER_TEMP_CHECK.begin(), TRIGGER_TEMP_CHECK.end(), TRIGGER_TEMP_CHECK.begin(), ::tolower);
    std::transform(TRIGGER_HELP.begin(), TRIGGER_HELP.end(), TRIGGER_HELP.begin(), ::tolower);
    std::transform(TRIGGER_DISCARD.begin(), TRIGGER_DISCARD.end(), TRIGGER_DISCARD.begin(), ::tolower);
    std::transform(TRIGGER_REPEAT.begin(), TRIGGER_REPEAT.end(), TRIGGER_REPEAT.begin(), ::tolower);
    std::transform(TRIGGER_STATUS.begin(), TRIGGER_STATUS.end(), TRIGGER_STATUS.begin(), ::tolower);
    std::transform(TRIGGER_PAUSE.begin(), TRIGGER_PAUSE.end(), TRIGGER_PAUSE.begin(), ::tolower);
    std::transform(TRIGGER_RESUME.begin(), TRIGGER_RESUME.end(), TRIGGER_RESUME.begin(), ::tolower);
    std::transform(TRIGGER_LIST_COMMANDS.begin(), TRIGGER_LIST_COMMANDS.end(), TRIGGER_LIST_COMMANDS.begin(), ::tolower);
    std::transform(TRIGGER_CAMERA_ON.begin(), TRIGGER_CAMERA_ON.end(), TRIGGER_CAMERA_ON.begin(), ::tolower);
    std::transform(TRIGGER_CAMERA_OFF.begin(), TRIGGER_CAMERA_OFF.end(), TRIGGER_CAMERA_OFF.begin(), ::tolower);
    OPENWEBUI_URL = ensure_trailing_slash(OPENWEBUI_URL);

    if (KNOWLEDGE_BASE_IDS.empty()) {
        say_error(tr(MSG_KB_NOT_SET));
    }

    return true;
}

void attach_configured_knowledge_collections(json& body) {
    const auto collection_ids = split_config_list(KNOWLEDGE_BASE_IDS);
    if (collection_ids.empty()) {
        return;
    }

    json files = json::array();
    for (const auto& id : collection_ids) {
        files.push_back({
            {"type", "collection"},
            {"id", id}
        });
    }
    body["files"] = std::move(files);
}

json build_chat_body(const std::string& user_content, bool enable_websearch, bool use_knowledge = true) {
    json body = {
        {"model", MODEL_NAME},
        {"messages", {
            {{"role", "system"}, {"content", "You are a helpful assistant."}},
            {{"role", "user"}, {"content", user_content}}
        }},
        {"stream", false},
        {"chat_id", ""},
        {"enable_websearch", enable_websearch}
    };

    if (use_knowledge) {
        attach_configured_knowledge_collections(body);
    }
    return body;
}

void initialize_openai_client() {
    std::call_once(openai_init_flag, [] {
        openai::start(API_KEY, "", true, OPENWEBUI_URL);
    });
}

// AI analysis with fresh context for each request
void analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex);
    const int analysis_id = ++counter_value;
    temp_counter_value = 0; // Reset temp counter for each main analysis
    help_counter_value = 0; // Reset help counter for each main analysis
    camera_counter_value = 0; // Reset camera counter for each main analysis
    say_info(tr(MSG_ANALYSIS_STARTED_PREFIX) + std::to_string(analysis_id) + tr(MSG_ANALYSIS_STARTED_SUFFIX));

    const std::string filename = "results_analysis" + std::to_string(analysis_id) + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error(tr(MSG_ERR_OPEN_RESULTS) + filename + "\n");
        say_info(tr(MSG_ANALYSIS_STARTED_PREFIX) + std::to_string(analysis_id) + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_HEADER) + filename + "\n");
    }

    std::string response_string;

    try {
        initialize_openai_client();

        json body = build_chat_body(PROMPT + "\n" + text, true);
        auto chat = openai::chat().create(body);
        const std::string api_error = extract_api_error(chat);
        if (!api_error.empty()) {
            throw std::runtime_error("OpenWebUI API error: " + api_error);
        }
        // Strip model-internal tags, then optionally run deterministic FHIR post-processing.
        response_string = strip_internal_reasoning_tags(extract_message_content(chat));
        if (check_fhir) {
            response_string = revise_fhir_bundle_in_response(response_string, std::to_string(analysis_id), file);
        }
        if (response_string.empty()) {
            file << "\n[WARN] No textual content found in primary response. Full payload:\n"
                 << chat.dump(2) << "\n";
            say_error(std::string{tr(MSG_WARN_NO_TEXT_PREFIX)} + std::to_string(analysis_id) +
                      tr(MSG_WARN_NO_TEXT_SUFFIX));
        }

        file << "\n\nFull response received:\n" << response_string << "\n";
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id << "] failed: " << e.what() << "\n";
        say_error(std::string{tr(MSG_ERR_ANALYSIS_FAILED_PREFIX)} + std::to_string(analysis_id) +
                  tr(MSG_ERR_ANALYSIS_FAILED_MIDDLE) + e.what() + "\n");
    }

    if (!response_string.empty() && !no_analysis_summary) {
        try {
            // Follow-up summary call keeps spoken output concise.
            json summary_body = build_chat_body(
                std::string{summary_prompt_for_language()} +
                    response_string + "\n\n",
                false,
                false);

            auto summary_chat = openai::chat().create(summary_body);
            const std::string summary_api_error = extract_api_error(summary_chat);
            if (!summary_api_error.empty()) {
                throw std::runtime_error("OpenWebUI API error: " + summary_api_error);
            }
            const std::string summary_string = strip_internal_reasoning_tags(extract_message_content(summary_chat));
            if (summary_string.empty()) {
                file << "\n[WARN] No textual summary returned. Full payload:\n"
                     << summary_chat.dump(2) << "\n";
                say_error(std::string{tr(MSG_WARN_SUMMARY_NO_TEXT_PREFIX)} + std::to_string(analysis_id) +
                          tr(MSG_WARN_SUMMARY_NO_TEXT_SUFFIX));
            }

            file << "\nShort summary of response:\n" << summary_string << "\n";
            speak_feedback(std::string{tr(MSG_ANALYSIS_FEEDBACK_PREFIX)} + std::to_string(analysis_id) +
                           tr(MSG_ANALYSIS_SUMMARY_FEEDBACK_MIDDLE) + summary_string);
        } catch (const std::exception& e) {
            file << "\n[ERROR] Summary generation failed: " << e.what() << "\n";
            say_error(std::string{tr(MSG_ERR_SUMMARY_FAILED_PREFIX)} + std::to_string(analysis_id) +
                      tr(MSG_ERR_SUMMARY_FAILED_MIDDLE) + e.what() + "\n");
        }
    }

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_RESULTS_PREFIX) + std::to_string(analysis_id) + tr(MSG_ERR_WRITE_RESULTS_SUFFIX));
    }

    say_info(tr(MSG_ANALYSIS_STARTED_PREFIX) + std::to_string(analysis_id) + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
}

void temp_analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex);
    const int analysis_id = ++temp_counter_value;
    // Use compound id (<main>.<temp>) so temp files sort with their parent analysis.
    const std::string analysis_id_str = std::to_string(counter_value + 1) + "." + std::to_string(analysis_id);
    say_info(tr(MSG_TEMP_ANALYSIS_STARTED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_STARTED_SUFFIX));

    const std::string filename = "tmp_results_analysis" + analysis_id_str + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error(tr(MSG_ERR_OPEN_RESULTS) + filename + "\n");
        say_info(tr(MSG_TEMP_ANALYSIS_FINISHED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << TEMP_PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_HEADER) + filename + "\n");
    }

    std::string response_string;

    try {
        initialize_openai_client();

        json body = build_chat_body(TEMP_PROMPT + "\n" + text, true);
        auto chat = openai::chat().create(body);
        const std::string api_error = extract_api_error(chat);
        if (!api_error.empty()) {
            throw std::runtime_error("OpenWebUI API error: " + api_error);
        }
        response_string = strip_internal_reasoning_tags(extract_message_content(chat));
        if (check_fhir) {
            response_string = revise_fhir_bundle_in_response(response_string, "tmp_" + analysis_id_str, file);
        }
        if (response_string.empty()) {
            file << "\n[WARN] No textual content found in temporary response. Full payload:\n"
                 << chat.dump(2) << "\n";
            say_error(std::string{tr(MSG_WARN_NO_TEXT_PREFIX)} + analysis_id_str +
                      tr(MSG_WARN_NO_TEXT_SUFFIX));
        }

        file << "\n\nTemporary response received:\n" << response_string << "\n";
        speak_feedback("Temporary Analysis[" + analysis_id_str + "] completed. Response: " + response_string);
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id_str << "] failed: " << e.what() << "\n";
        say_error(std::string{tr(MSG_ERR_ANALYSIS_FAILED_PREFIX)} + analysis_id_str +
                  tr(MSG_ERR_ANALYSIS_FAILED_MIDDLE) + e.what() + "\n");
    }

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_RESULTS_PREFIX) + analysis_id_str + tr(MSG_ERR_WRITE_RESULTS_SUFFIX));
    }

    say_info(tr(MSG_TEMP_ANALYSIS_FINISHED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
}

void help_analyze_text(const std::string& text) {
    AnalysisSession session(analysis_mutex);
    const int analysis_id = ++help_counter_value;
    // Use compound id (<main>.<help>) so help files sort with their parent analysis.
    const std::string analysis_id_str = std::to_string(counter_value + 1) + "." + std::to_string(analysis_id);
    say_info(tr(MSG_HELP_ANALYSIS_STARTED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_STARTED_SUFFIX));

    const std::string filename = "tmp_help_analysis" + analysis_id_str + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        say_error(tr(MSG_ERR_OPEN_RESULTS) + filename + "\n");
        say_info(tr(MSG_HELP_ANALYSIS_FINISHED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
        return;
    }

    file << "Using model: " << MODEL_NAME << "\n";
    file << "Endpoint: " << OPENWEBUI_URL << "\n";
    file << "Prompt: " << HELP_PROMPT << "\n" << text << "\n";

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_HEADER) + filename + "\n");
    }

    std::string response_string;

    try {
        initialize_openai_client();

        json body = build_chat_body(HELP_PROMPT + "\n" + text, true);
        auto chat = openai::chat().create(body);
        const std::string api_error = extract_api_error(chat);
        if (!api_error.empty()) {
            throw std::runtime_error("OpenWebUI API error: " + api_error);
        }
        response_string = strip_internal_reasoning_tags(extract_message_content(chat));
        if (check_fhir) {
            response_string = revise_fhir_bundle_in_response(response_string, "help_" + analysis_id_str, file);
        }
        if (response_string.empty()) {
            file << "\n[WARN] No textual content found in help response. Full payload:\n"
                 << chat.dump(2) << "\n";
            say_error(std::string{tr(MSG_WARN_NO_TEXT_PREFIX)} + analysis_id_str +
                      tr(MSG_WARN_NO_TEXT_SUFFIX));
        }

        file << "\n\nHelp response received:\n" << response_string << "\n";
        speak_feedback("Help[" + analysis_id_str + "] completed. Suggestion: " + response_string);
    } catch (const std::exception& e) {
        file << "\n[ERROR] Analysis[" << analysis_id_str << "] failed: " << e.what() << "\n";
        say_error(std::string{tr(MSG_ERR_ANALYSIS_FAILED_PREFIX)} + analysis_id_str +
                  tr(MSG_ERR_ANALYSIS_FAILED_MIDDLE) + e.what() + "\n");
    }

    if (!file) {
        say_error(tr(MSG_ERR_WRITE_RESULTS_PREFIX) + analysis_id_str + tr(MSG_ERR_WRITE_RESULTS_SUFFIX));
    }

    say_info(tr(MSG_HELP_ANALYSIS_FINISHED_PREFIX) + analysis_id_str + tr(MSG_ANALYSIS_FINISHED_SUFFIX));
}

using AnalysisFunction = void (*)(const std::string&);

bool launch_analysis_thread(AnalysisFunction function, std::string text) {
    ++active_analyses;

    try {
        std::thread([function, text = std::move(text)]() mutable {
            AnalysisJobGuard guard(active_analyses);
            function(text);
        }).detach();
        return true;
    } catch (const std::exception& e) {
        --active_analyses;
        say_error(std::string{"[ERROR] Failed to start analysis thread: "} + e.what() + "\n");
        return false;
    }
}

static void print_help(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [OPTIONS]\n"
        "\n"
        "Listens on standard input for trigger words, collects spoken text between\n"
        "start/stop triggers, and sends it to a configured AI model for analysis.\n"
        "Results are written to numbered files (results_analysis<N>.txt).\n"
        "\n"
        "Options:\n"
        "  --help                  Show this help message and exit.\n"
        "  --check_fhir            After each analysis, detect and post-process any\n"
        "                          FHIR Bundle found in the model response using the\n"
        "                          deterministic_fhir_mapper tool.\n"
        "  --no_analysis_summary   Skip analysis summary generation and spoken\n"
        "                          analysis-summary feedback.\n"
        "  --language <lang>       Set the UI language for console messages.\n"
        "                          Supported values: en (default), it, fr. Summary\n"
        "                          feedback uses the selected language.\n"
        "\n"
        "Configuration:\n"
        "  The program reads ./config.ini on startup. The following sections and\n"
        "  keys are required:\n"
        "    [openai]   base_url, api_key, model_name\n"
        "    [prompts]  prompt, temp_prompt, help_prompt\n"
        "    [triggers] start, stop, temp_check, help, discard, repeat, status,\n"
        "               pause, resume, list_commands, camera_on, camera_off\n"
        "    [tts]      command\n"
        "  Optional keys:\n"
        "    [analysis]             knowledge_base_ids\n"
        "    [deterministic_mapper] network_enabled, cache_dir, cache_ttl_days,\n"
        "                           loinc_user, loinc_pass, timeout_seconds\n"
        "    [tts]                  self_echo_grace_seconds\n"
        "    [camera]               camera_id, camera_interval\n"
        "\n"
        "Trigger words (configured in config.ini):\n"
        "  start      Begin collecting transcribed speech.\n"
        "  stop       Stop collecting and send text to AI for full analysis.\n"
        "             If an analysis is already running, recording continues and\n"
        "             stop must be requested again after it finishes.\n"
        "  temp_check Perform a temporary analysis on a snapshot of collected text\n"
        "             without stopping the recording.\n"
        "  help       Ask the AI for a suggestion on what to do next, based on a\n"
        "             snapshot of the collected text so far, without stopping the\n"
        "             recording.\n"
        "  discard    Discard the currently collected text and stop recording,\n"
        "             without sending anything to the AI.\n"
        "  pause      Temporarily stop appending speech to the collected text,\n"
        "             without discarding what has been collected so far.\n"
        "  resume     Resume appending speech after a pause.\n"
        "  repeat     Speak the last substantive feedback again (analysis summary,\n"
        "             temporary-check response, or help suggestion).\n"
        "  status     Report whether recording is on, off, or paused, and how many\n"
        "             analyses are currently running.\n"
        "  list_commands  Speak the list of all configured voice command phrases.\n"
        "  camera_on      Launch realtime_video_pipeline.exe in the background,\n"
        "                 without stopping the recording. Requires an active\n"
        "                 recording session, like temp_check.\n"
        "  camera_off     Kill any running realtime_video_pipeline.exe process,\n"
        "                 without stopping the recording. Requires an active\n"
        "                 recording session, like temp_check.\n"
        "\n"
        "Exit status:\n"
        "  0  Normal exit (EOF on stdin).\n"
        "  1  Configuration error or invalid argument.\n";
}

// Main loop
int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--check_fhir") {
            check_fhir = true;
        } else if (arg == "--no_analysis_summary" || arg == "--no-analysis-summary") {
            no_analysis_summary = true;
        } else if (arg == "--language") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --language requires an argument. Supported: en, it, fr\n";
                return 1;
            }
            const std::string lang(argv[++i]);
            if (lang == "it") {
                g_lang = Lang::IT;
            } else if (lang == "fr") {
                g_lang = Lang::FR;
            } else if (lang != "en") {
                std::cerr << "Error: unknown language '" << lang << "'. Supported: en, it, fr\n";
                return 1;
            }
        } else {
            std::cerr << "Error: unknown option '" << arg << "'. Use --help for usage information.\n";
            return 1;
        }
    }

    if (!load_config("./config.ini")) {
        say_error(tr(MSG_FAILED_LOAD_CONFIG));
        return 1;
    }

    say_info(tr(MSG_LISTENING));

    std::string line;
    std::string collected_text;
    RecordingState recording_state = RecordingState::Idle;
    bool camera_running = false;

    while (std::getline(std::cin, line)) {
        std::cout << line << std::endl;

        if (is_self_echo_muted()) {
            // Still within the window where the mic may be picking up our
            // own list_commands announcement; ignore it rather than acting
            // on it or appending it to the transcript.
            continue;
        }

        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);

        const bool line_contains_start = contains_substring(lower_line, TRIGGER_START);
        const bool line_contains_stop = contains_substring(lower_line, TRIGGER_STOP);
        const bool line_contains_temp_check = contains_substring(lower_line, TRIGGER_TEMP_CHECK);
        const bool line_contains_help = contains_substring(lower_line, TRIGGER_HELP);
        const bool line_contains_discard = contains_substring(lower_line, TRIGGER_DISCARD);
        const bool line_contains_repeat = contains_substring(lower_line, TRIGGER_REPEAT);
        const bool line_contains_status = contains_substring(lower_line, TRIGGER_STATUS);
        const bool line_contains_pause = contains_substring(lower_line, TRIGGER_PAUSE);
        const bool line_contains_resume = contains_substring(lower_line, TRIGGER_RESUME);
        const bool line_contains_list_commands = contains_substring(lower_line, TRIGGER_LIST_COMMANDS);
        const bool line_contains_camera_on = contains_substring(lower_line, TRIGGER_CAMERA_ON);
        const bool line_contains_camera_off = contains_substring(lower_line, TRIGGER_CAMERA_OFF);

        if (line_contains_start) {
            if (recording_state != RecordingState::Idle) {
                say_info(tr(MSG_RECORDING_ALREADY_STARTED));
            } else {
                say_info(tr(MSG_RECORDING_STARTED));
                collected_text.clear();
                recording_state = RecordingState::Collecting;
            }
        }

        if (line_contains_stop) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else if (active_analyses.load() > 0) {
                say_info(tr(MSG_ANALYSIS_RUNNING_STOP_BLOCKED));
            } else {
                say_info(tr(MSG_RECORDING_STOPPED));
                std::string text_to_analyze = collected_text;
                collected_text.clear();
                recording_state = RecordingState::Idle;
                launch_analysis_thread(analyze_text, std::move(text_to_analyze));
            }
        }

        if (line_contains_temp_check) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else {
                say_info(tr(MSG_TEMP_CHECK_REQUESTED));
                if (active_analyses.load() > 0) {
                    say_info(tr(MSG_ANOTHER_ANALYSIS_RUNNING));
                }
                std::string snapshot = collected_text;
                // Temp analysis runs on a snapshot while recording continues.
                launch_analysis_thread(temp_analyze_text, std::move(snapshot));
            }
        }

        if (line_contains_camera_on) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else {
                say_info(tr(MSG_CAMERA_ON_REQUESTED));
                const int camera_analysis_id = ++camera_counter_value;
                // Use compound id (<main>.<camera>) so video files sort with
                // their parent analysis, exactly like tmp_help_analysis<N>.txt.
                const std::string camera_id_str = std::to_string(counter_value + 1) + "." +
                                                   std::to_string(camera_analysis_id);
                const std::string video_filename = "video_analysis" + camera_id_str + ".txt";

                std::ostringstream camera_cmd;
                camera_cmd << "./realtime_video_pipeline.exe '" << escape_for_single_quotes(CAMERA_ID)
                           << "' --interval '" << escape_for_single_quotes(CAMERA_INTERVAL)
                           << "' > '" << escape_for_single_quotes(video_filename) << "' &";
                std::system(camera_cmd.str().c_str());
                camera_running = true;
            }
        }

        if (line_contains_camera_off) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else {
                say_info(tr(MSG_CAMERA_OFF_REQUESTED));
                std::system("killall realtime_video_pipeline.exe >/dev/null 2>&1");
                camera_running = false;
            }
        }

        if (line_contains_help) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else {
                say_info(tr(MSG_HELP_REQUESTED));
                if (active_analyses.load() > 0) {
                    say_info(tr(MSG_ANOTHER_ANALYSIS_RUNNING));
                }
                std::string snapshot = collected_text;
                // Help analysis runs on a snapshot while recording continues.
                launch_analysis_thread(help_analyze_text, std::move(snapshot));
            }
        }

        if (line_contains_discard) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else {
                collected_text.clear();
                recording_state = RecordingState::Idle;
                say_info(tr(MSG_RECORDING_DISCARDED));
            }
        }

        if (line_contains_pause) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else if (recording_state == RecordingState::Paused) {
                say_info(tr(MSG_RECORDING_ALREADY_PAUSED));
            } else {
                recording_state = RecordingState::Paused;
                say_info(tr(MSG_RECORDING_PAUSED));
            }
        }

        if (line_contains_resume) {
            if (recording_state == RecordingState::Idle) {
                say_info(tr(MSG_NO_RECORDING_RUNNING));
            } else if (recording_state == RecordingState::Collecting) {
                say_info(tr(MSG_RECORDING_NOT_PAUSED));
            } else {
                recording_state = RecordingState::Collecting;
                say_info(tr(MSG_RECORDING_RESUMED));
            }
        }

        if (line_contains_repeat) {
            std::string feedback_copy;
            {
                std::lock_guard<std::mutex> lock(feedback_mutex);
                feedback_copy = last_feedback_message;
            }
            if (feedback_copy.empty()) {
                say_info(tr(MSG_NOTHING_TO_REPEAT));
            } else {
                // Replay audio only, mirroring how the original feedback was delivered.
                speak_text(feedback_copy);
            }
        }

        if (line_contains_status) {
            const char* state_str = (recording_state == RecordingState::Collecting) ? tr(MSG_STATUS_STATE_COLLECTING)
                                   : (recording_state == RecordingState::Paused) ? tr(MSG_STATUS_STATE_PAUSED)
                                   : tr(MSG_STATUS_STATE_IDLE);
            const char* camera_state_str = camera_running ? tr(MSG_STATUS_CAMERA_ON) : tr(MSG_STATUS_CAMERA_OFF);
            std::ostringstream status_oss;
            status_oss << tr(MSG_STATUS_HEADER) << state_str << tr(MSG_STATUS_ANALYSES_MIDDLE)
                       << active_analyses.load() << tr(MSG_STATUS_CAMERA_MIDDLE) << camera_state_str << ".\n";
            say_info(status_oss.str());
        }

        if (line_contains_list_commands) {
            std::ostringstream commands_oss;
            commands_oss << tr(MSG_LIST_COMMANDS_HEADER)
                         << TRIGGER_START << ", " << TRIGGER_STOP << ", " << TRIGGER_TEMP_CHECK << ", "
                         << TRIGGER_HELP << ", " << TRIGGER_DISCARD << ", " << TRIGGER_REPEAT << ", "
                         << TRIGGER_STATUS << ", " << TRIGGER_PAUSE << ", " << TRIGGER_RESUME << ", "
                         << TRIGGER_CAMERA_ON << ", " << TRIGGER_CAMERA_OFF << ", "
                         << TRIGGER_LIST_COMMANDS << ".\n";
            const std::string commands_message = commands_oss.str();
            std::cout << commands_message;
            // The response recites every trigger phrase verbatim. Block here
            // until playback has actually finished (rather than estimating
            // how long it will take) so the main loop isn't reading stdin
            // while the mic could still be hearing it, then hold a short
            // extra grace period for ASR pipeline latency.
            speak_text(commands_message, /*wait_for_completion=*/true);
            mute_self_echo_for(std::chrono::milliseconds(
                static_cast<int64_t>(SELF_ECHO_GRACE_SECONDS * 1000)));
        }

        if (recording_state == RecordingState::Collecting && !line_contains_start && !line_contains_stop &&
            !line_contains_temp_check && !line_contains_help && !line_contains_discard &&
            !line_contains_pause && !line_contains_resume && !line_contains_repeat && !line_contains_status &&
            !line_contains_list_commands && !line_contains_camera_on && !line_contains_camera_off) {
            collected_text += line + "\n";
        }
    }

    return 0;
}
