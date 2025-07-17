// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Speech-to-Data project.
//
// Copyright (C) 2025 Spazio IT 
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// This file is based on https://github.com/davabase/whisper_real_time, with modifications
// for real-time performance, concurrency improvements, and debug tracing.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see https://www.gnu.org/licenses/.
//
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cctype>
#include <atomic>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <ctime>
// PortAudio includes
#include "portaudio.h"
// whisper.cpp includes
#include "whisper.h"

#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 1024
#define MIN_AUDIO_LENGTH_MS 100
#define MIN_AUDIO_SAMPLES static_cast<size_t>(SAMPLE_RATE * MIN_AUDIO_LENGTH_MS / 1000.0)

struct Args {
    std::string model = "medium";
    bool non_english = false;
    int energy_threshold = -1;
    double record_timeout = 2.0;
    double phrase_timeout = 3.0;
    std::string language = "en";
    bool pipe = false;
    bool timestamp = false;
    std::string default_microphone;
    std::string whisper_model_path;
};

class AudioRecorder {
public:
    virtual ~AudioRecorder() = default;
    virtual bool startRecording(std::function<void(const std::vector<int16_t>&)> callback, 
                               int sampleRate, 
                               double recordTimeout, 
                               double phraseTimeout) = 0;
    virtual void stopRecording() = 0;
    virtual void adjustForAmbientNoise(int energy_threshold) = 0;
    virtual void setEnergyThreshold(int threshold) = 0;
    virtual int getEnergyThreshold() const = 0;
    static std::vector<std::string> listMicrophoneNames();
};

class PortAudioRecorder : public AudioRecorder {
private:
    PaStream *stream = nullptr;
    std::function<void(const std::vector<int16_t>&)> audioCallback;
    std::atomic<bool> recordingActive{false};
    std::atomic<int> energyThreshold{1000};
    int sampleRate_ = SAMPLE_RATE;
    double recordTimeout_ = 2.0;
    double phraseTimeout_ = 3.0;
    size_t max_buffer_samples_ = 0;
    size_t max_silence_chunks_ = 0;
    std::atomic<bool> bypass_vad_{false};
    
    // VAD state
    std::vector<int16_t> vad_buffer;
    std::mutex vad_buffer_mutex;
    size_t consecutive_silence_chunks_ = 0;
    
    // Callback protection
    std::mutex callback_mutex_;
    
    // PortAudio initialization flag
    static std::atomic<bool> pa_initialized;
    static std::mutex pa_init_mutex;
    
    static int pa_callback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData);

    void ensure_pa_initialized() {
        std::lock_guard<std::mutex> lock(pa_init_mutex);
        if (!pa_initialized) {
            PaError err = Pa_Initialize();
            if (err != paNoError) {
                throw std::runtime_error("PortAudio init failed: " + std::string(Pa_GetErrorText(err)));
            }
            pa_initialized = true;
            std::atexit([]() {
                if (pa_initialized) {
                    Pa_Terminate();
                    pa_initialized = false;
                }
            });
        }
    }

public:
    PortAudioRecorder();
    ~PortAudioRecorder();
    bool startRecording(std::function<void(const std::vector<int16_t>&)> callback, 
                       int sampleRate, 
                       double recordTimeout,
                       double phraseTimeout) override;
    void stopRecording() override;
    void adjustForAmbientNoise(int energy_threshold) override;
    void setEnergyThreshold(int threshold) override { energyThreshold = threshold; }
    int getEnergyThreshold() const override { return energyThreshold.load(); }
    static void setInizialized(bool initialized) {
        pa_initialized = initialized;
    }
    static bool isInitialized() {
        return pa_initialized.load();
    }
    static void ensureInitialized() {
        std::lock_guard<std::mutex> lock(pa_init_mutex);
        if (!pa_initialized) {
            PaError err = Pa_Initialize();
            if (err != paNoError) {
                throw std::runtime_error("PortAudio initialization failed: " + std::string(Pa_GetErrorText(err)));
            }
            pa_initialized = true;
        }
    }
};

// Initialize static members
std::atomic<bool> PortAudioRecorder::pa_initialized{false};
std::mutex PortAudioRecorder::pa_init_mutex;

std::vector<std::string> AudioRecorder::listMicrophoneNames() {
    std::vector<std::string> names;
    PortAudioRecorder::ensureInitialized();
    
    if (!PortAudioRecorder::isInitialized()) {
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "PortAudio error (listMicrophoneNames init): " << Pa_GetErrorText(err) << std::endl;
            return names;
        }
        PortAudioRecorder::setInizialized(true);
    }

    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "PortAudio error (listMicrophoneNames device count): " << Pa_GetErrorText(numDevices) << std::endl;
        return names;
    }

    const PaDeviceInfo *deviceInfo;
    for (int i = 0; i < numDevices; ++i) {
        deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            names.push_back(deviceInfo->name);
        }
    }
    return names;
}

PortAudioRecorder::PortAudioRecorder() {
    ensure_pa_initialized();
}

PortAudioRecorder::~PortAudioRecorder() {
    stopRecording();
    if (stream) {
        Pa_CloseStream(stream);
        stream = nullptr;
    }
}

int PortAudioRecorder::pa_callback(const void *inputBuffer, void *outputBuffer,
                                   unsigned long framesPerBuffer,
                                   const PaStreamCallbackTimeInfo* timeInfo,
                                   PaStreamCallbackFlags statusFlags,
                                   void *userData) {
    PortAudioRecorder *recorder = static_cast<PortAudioRecorder*>(userData);
    const int16_t *in = static_cast<const int16_t*>(inputBuffer);
    if (inputBuffer == nullptr) {
        return paContinue;
    }

    // Check if we should bypass VAD
    bool bypass = recorder->bypass_vad_.load();
    std::function<void(const std::vector<int16_t>&)> callback;
    {
        std::lock_guard<std::mutex> lock(recorder->callback_mutex_);
        callback = recorder->audioCallback;
    }

    if (bypass) {
        std::vector<int16_t> chunk(in, in + framesPerBuffer);
        if (callback) {
            callback(chunk);
        }
        return paContinue;
    }

    std::vector<int16_t> current_chunk(in, in + framesPerBuffer);

    // Calculate RMS for VAD
    double sum_squares = 0.0;
    for (int16_t sample : current_chunk) {
        sum_squares += static_cast<double>(sample) * sample;
    }
    double rms = std::sqrt(sum_squares / framesPerBuffer);
    int threshold = recorder->getEnergyThreshold();

    {
        std::lock_guard<std::mutex> lock(recorder->vad_buffer_mutex);
        if (rms > threshold) {
            // Reset silence counter on voice activity
            recorder->consecutive_silence_chunks_ = 0;
            recorder->vad_buffer.insert(recorder->vad_buffer.end(), current_chunk.begin(), current_chunk.end());
        } else {
            // Only accumulate silence if we're in a speech segment
            if (!recorder->vad_buffer.empty()) {
                recorder->consecutive_silence_chunks_++;
                recorder->vad_buffer.insert(recorder->vad_buffer.end(), current_chunk.begin(), current_chunk.end());
            }
        }

        // Send buffer if max size reached or enough trailing silence
        if (!recorder->vad_buffer.empty() && 
            (recorder->vad_buffer.size() >= recorder->max_buffer_samples_ || 
             recorder->consecutive_silence_chunks_ >= recorder->max_silence_chunks_)) {
            if (callback) {
                callback(recorder->vad_buffer);
            }
            recorder->vad_buffer.clear();
            recorder->consecutive_silence_chunks_ = 0;
        }
    }
    return paContinue;
}

bool PortAudioRecorder::startRecording(std::function<void(const std::vector<int16_t>&)> callback, 
                                      int sampleRate, 
                                      double recordTimeout,
                                      double phraseTimeout) {
    if (recordingActive) {
        stopRecording();
    }

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        audioCallback = callback;
    }
    
    sampleRate_ = sampleRate;
    recordTimeout_ = recordTimeout;
    phraseTimeout_ = phraseTimeout;
    
    // Calculate buffer limits
    max_buffer_samples_ = static_cast<size_t>(sampleRate_ * recordTimeout_);
    max_silence_chunks_ = static_cast<size_t>(std::ceil(phraseTimeout_ * sampleRate_ / FRAMES_PER_BUFFER));
    consecutive_silence_chunks_ = 0;
    bypass_vad_ = false;
    recordingActive = true;

    ensure_pa_initialized();

    PaStreamParameters inputParameters;
    inputParameters.device = Pa_GetDefaultInputDevice();
    if (inputParameters.device == paNoDevice) {
        std::cerr << "Error: No default input device." << std::endl;
        return false;
    }
    
    inputParameters.channelCount = 1;
    inputParameters.sampleFormat = paInt16;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(
        &stream,
        &inputParameters,
        nullptr,
        sampleRate_,
        FRAMES_PER_BUFFER,
        paClipOff,
        pa_callback,
        this
    );

    if (err != paNoError) {
        std::cerr << "PortAudio error (open stream): " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error (start stream): " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        stream = nullptr;
        return false;
    }

    std::cout << "Started recording on: " << Pa_GetDeviceInfo(inputParameters.device)->name << std::endl;
    return true;
}

void PortAudioRecorder::stopRecording() {
    if (recordingActive) {
        recordingActive = false;
        if (stream) {
            PaError err = Pa_StopStream(stream);
            if (err != paNoError) {
                std::cerr << "PortAudio error (stop stream): " << Pa_GetErrorText(err) << std::endl;
            }
            err = Pa_CloseStream(stream);
            if (err != paNoError) {
                std::cerr << "PortAudio error (close stream): " << Pa_GetErrorText(err) << std::endl;
            }
            stream = nullptr;
        }
        
        // Clear any remaining VAD buffer
        std::lock_guard<std::mutex> lock(vad_buffer_mutex);
        vad_buffer.clear();
    }
}

void PortAudioRecorder::adjustForAmbientNoise(int energy_threshold) {
    if (energy_threshold != -1) {
        energyThreshold = energy_threshold;
        std::cout << "Using provided energy threshold: " << energy_threshold << std::endl;
        return;
    }

    std::cout << "Adjusting for ambient noise (listening for 3 seconds)..." << std::endl;
    std::vector<int16_t> noise_samples;
    std::mutex noise_mutex;
    std::condition_variable noise_cv;
    bool noise_collection_done = false;

    // Create temporary callback
    auto noise_callback = [&](const std::vector<int16_t>& audio_data) {
        std::lock_guard<std::mutex> lock(noise_mutex);
        noise_samples.insert(noise_samples.end(), audio_data.begin(), audio_data.end());
        if (noise_samples.size() >= static_cast<size_t>(SAMPLE_RATE * 3)) {
            noise_collection_done = true;
            noise_cv.notify_one();
        }
    };

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        audioCallback = noise_callback;
    }
    bypass_vad_ = true;

    // Start the stream if not already running
    bool was_running = recordingActive;
    if (!was_running) {
        if (!startRecording(noise_callback, sampleRate_, recordTimeout_, phraseTimeout_)) {
            std::cerr << "Failed to start recording for ambient noise adjustment." << std::endl;
            bypass_vad_ = false;
            return;
        }
    }

    // Wait for noise collection to complete
    std::unique_lock<std::mutex> lock(noise_mutex);
    auto status = noise_cv.wait_for(lock, std::chrono::seconds(4), [&]{ return noise_collection_done; });

    bypass_vad_ = false;
    if (!was_running) {
        stopRecording();
    }

    if (noise_samples.empty()) {
        std::cerr << "No noise samples collected. Using default energy threshold." << std::endl;
        return;
    }

    // Calculate RMS of noise samples
    double sum_squares = 0.0;
    for (int16_t sample : noise_samples) {
        sum_squares += static_cast<double>(sample) * sample;
    }
    double rms = std::sqrt(sum_squares / noise_samples.size());
    
    // Set threshold based on estimated noise RMS
    energyThreshold = static_cast<int>(rms * 2.5);
    std::cout << "Adjusted energy threshold to: " << energyThreshold << std::endl;
}

class WhisperModel {
private:
    whisper_context *ctx = nullptr;
    std::string model_path;
    
public:
    WhisperModel(const std::string& modelPath) : model_path(modelPath) {
        if (!std::filesystem::exists(modelPath)) {
            throw std::runtime_error("Model file does not exist: " + modelPath);
        }
        
        std::cout << "Loading Whisper model from: " << model_path << std::endl;
        ctx = whisper_init_from_file(model_path.c_str());
        if (!ctx) {
            throw std::runtime_error("Failed to load Whisper model from " + model_path);
        }
    }
    
    ~WhisperModel() {
        if (ctx) {
            whisper_free(ctx);
        }
    }
    
    std::string transcribe(const std::vector<float>& audio_data_normalized, const std::string& lang) {
        if (!ctx || audio_data_normalized.empty()) {
            return "";
        }
        
        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.language = lang.c_str();
        params.n_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
        params.print_realtime = false;
        params.print_progress = false;
        params.print_timestamps = false;
        params.single_segment = true;
        
        if (whisper_full(ctx, params, audio_data_normalized.data(), audio_data_normalized.size()) != 0) {
            return "";
        }
        
        std::string result_text;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char *text = whisper_full_get_segment_text(ctx, i);
            if (text) {
                result_text += text;
            }
        }
        return result_text;
    }
};

Args parse_arguments(int argc, char* argv[]) {
    Args args;
    const std::unordered_set<std::string> valid_args = {
        "--model", "--non_english", "--energy_threshold", "--record_timeout",
        "--phrase_timeout", "--language", "--pipe", "--default_microphone",
        "--whisper_model_path", "--help", "-h", "--timestamp"
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (valid_args.find(arg) == valid_args.end()) {
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
            exit(1);
        }
        
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--non_english") {
            args.non_english = true;
        } else if (arg == "--energy_threshold" && i + 1 < argc) {
            args.energy_threshold = std::stoi(argv[++i]);
        } else if (arg == "--record_timeout" && i + 1 < argc) {
            args.record_timeout = std::stod(argv[++i]);
            if (args.record_timeout <= 0) {
                std::cerr << "Error: record_timeout must be positive" << std::endl;
                exit(1);
            }
        } else if (arg == "--phrase_timeout" && i + 1 < argc) {
            args.phrase_timeout = std::stod(argv[++i]);
            if (args.phrase_timeout <= 0) {
                std::cerr << "Error: phrase_timeout must be positive" << std::endl;
                exit(1);
            }
        } else if (arg == "--language" && i + 1 < argc) {
            args.language = argv[++i];
        } else if (arg == "--pipe") {
            args.pipe = true;
        } else if (arg == "--timestamp") {
            args.timestamp = true;
        } else if (arg == "--default_microphone" && i + 1 < argc) {
            args.default_microphone = argv[++i];
        } else if (arg == "--whisper_model_path" && i + 1 < argc) {
            args.whisper_model_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model <name>            Model to use (tiny, base, small, medium, large). Default: medium\n"
                      << "  --non_english             Don't use the English-specific model variant.\n"
                      << "  --energy_threshold <int>  Energy level for mic to detect. Default: auto-adjust\n"
                      << "  --record_timeout <float>  Max duration for audio chunks (seconds). Default: 2.0\n"
                      << "  --phrase_timeout <float>  Silence duration to end a phrase (seconds). Default: 3.0\n"
                      << "  --language <lang>         Language for transcription (de, en, es, fr, he, it, sv). Default: en\n"
                      << "  --pipe                    Enable pipe mode for continuous streaming.\n"
                      << "  --timestamp               Print timestamps before each line in pipe mode.\n"
                      << "  --whisper_model_path <path> REQUIRED: Path to the ggml Whisper model file\n";
            #ifdef __linux__
            std::cout << "  --default_microphone <name> Default microphone name. Use 'list' to see options.\n";
            #endif
            exit(0);
        }
    }

    if (args.whisper_model_path.empty()) {
        std::cerr << "Error: --whisper_model_path is required." << std::endl;
        exit(1);
    }

    return args;
}

void clear_console() {
    #ifdef _WIN32
        system("cls");
    #else
        std::cout << "\033[2J\033[H";
    #endif
}

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_tm = *std::localtime(&now_time_t);
    
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "[%Y-%m-%d %H:%M:%S]");
    return oss.str();
}

int main(int argc, char* argv[]) {
    Args args = parse_arguments(argc, argv);
    std::chrono::time_point<std::chrono::system_clock> last_phrase_end_time;
    bool phrase_time_set = false;
    std::queue<std::vector<int16_t>> data_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    std::unique_ptr<PortAudioRecorder> recorder;
    try {
        recorder = std::make_unique<PortAudioRecorder>();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize recorder: " << e.what() << std::endl;
        return 1;
    }

    WhisperModel audio_model(args.whisper_model_path);
    std::vector<std::string> transcription = {""};

    // Calibrate microphone if energy threshold not set
    if (args.energy_threshold == -1) {
        std::cout << "Calibrating microphone..." << std::endl;
        recorder->adjustForAmbientNoise(args.energy_threshold);
    } else {
        recorder->setEnergyThreshold(args.energy_threshold);
        std::cout << "Using energy threshold: " << args.energy_threshold << std::endl;
    }

    // Start continuous recording
    auto record_callback = [&](const std::vector<int16_t>& audio_data) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        data_queue.push(audio_data);
        queue_cv.notify_one();
    };

    if (!recorder->startRecording(record_callback, SAMPLE_RATE, args.record_timeout, args.phrase_timeout)) {
        std::cerr << "Failed to start continuous recording." << std::endl;
        return 1;
    }

    if (!args.pipe) {
        std::cout << "Model loaded and recording started.\n" << std::endl;
    }

    try {
        while (true) {
            std::vector<int16_t> audio_data;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (queue_cv.wait_for(lock, std::chrono::milliseconds(250), [&]{ return !data_queue.empty(); })) {
                    audio_data = std::move(data_queue.front());
                    data_queue.pop();
                }
            }

            auto now = std::chrono::system_clock::now();
            bool phrase_complete = false;
            if (phrase_time_set && (now - last_phrase_end_time) > std::chrono::duration<double>(args.phrase_timeout)) {
                phrase_complete = true;
            }

            if (!audio_data.empty()) {
                last_phrase_end_time = now;
                phrase_time_set = true;

                // Pad audio to minimum length if needed
                if (audio_data.size() < MIN_AUDIO_SAMPLES) {
                    audio_data.resize(MIN_AUDIO_SAMPLES, 0);
                }

                // Convert to float and normalize
                std::vector<float> audio_np(audio_data.size());
                for (size_t i = 0; i < audio_data.size(); ++i) {
                    audio_np[i] = static_cast<float>(audio_data[i]) / 32768.0f;
                }

                std::string text = audio_model.transcribe(audio_np, args.language);
                text = trim(text);

                // Skip empty transcriptions
                if (text.empty()) {
                    continue;
                }

                if (args.pipe) {
                    if (args.timestamp) {
                        std::cout << get_current_timestamp() << " " << text << std::endl;
                    } else {
                        std::cout << text << std::endl;
                    }
                } else {
                    if (phrase_complete) {
                        transcription.push_back(text);
                    } else {
                        transcription.back() = text;
                    }
                    clear_console();
                    for (const auto& line : transcription) {
                        if (!line.empty()) {
                            std::cout << line << std::endl;
                        }
                    }
                    std::cout << std::flush;
                }
            } else {
                if (phrase_time_set && (now - last_phrase_end_time) > std::chrono::duration<double>(args.phrase_timeout * 1.5)) {
                    if (!transcription.back().empty()) {
                        transcription.push_back("");
                    }
                    phrase_time_set = false;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }

    recorder->stopRecording();
    if (!args.pipe && !transcription.empty()) {
        std::cout << "\n\nTranscription:" << std::endl;
        for (const auto& line : transcription) {
            if (!line.empty()) {
                std::cout << line << std::endl;
            }
        }
    }
    return 0;
}