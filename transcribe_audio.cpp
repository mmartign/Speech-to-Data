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
#include <numeric> // For std::accumulate
#include <cmath>   // For std::sqrt
#include <cctype>  // For std::isspace
#include <atomic>

// PortAudio includes
#include "portaudio.h"

// whisper.cpp includes
#include "whisper.h"

#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 1024
#define MIN_AUDIO_LENGTH_MS 100
#define MIN_AUDIO_SAMPLES static_cast<size_t>(SAMPLE_RATE * MIN_AUDIO_LENGTH_MS / 1000.0)

class AudioRecorder {
public:
    virtual ~AudioRecorder() = default;
    virtual bool startRecording(std::function<void(const std::vector<int16_t>&)> callback, 
                               int sampleRate, 
                               double recordTimeout, 
                               double phraseTimeout) = 0;
    virtual void stopRecording() = 0;
    virtual void adjustForAmbientNoise() = 0;
    virtual void setEnergyThreshold(int threshold) = 0;
    virtual int getEnergyThreshold() const = 0;
    static std::vector<std::string> listMicrophoneNames();
};

class PortAudioRecorder : public AudioRecorder {
private:
    PaStream *stream;
    std::function<void(const std::vector<int16_t>&)> audioCallback;
    std::atomic<bool> recordingActive;
    int energyThreshold;
    int sampleRate_;
    double recordTimeout_;
    double phraseTimeout_;
    size_t max_buffer_samples_;
    size_t max_silence_chunks_;
    std::atomic<bool> bypass_vad_;

    // VAD state
    std::vector<int16_t> vad_buffer;
    std::mutex vad_buffer_mutex;
    size_t consecutive_silence_chunks_;

    static int pa_callback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData);

public:
    PortAudioRecorder();
    ~PortAudioRecorder();

    bool startRecording(std::function<void(const std::vector<int16_t>&)> callback, 
                       int sampleRate, 
                       double recordTimeout,
                       double phraseTimeout) override;
    void stopRecording() override;
    void adjustForAmbientNoise() override;
    void setEnergyThreshold(int threshold) override { energyThreshold = threshold; }
    int getEnergyThreshold() const override { return energyThreshold; }
};

std::vector<std::string> AudioRecorder::listMicrophoneNames() {
    std::vector<std::string> names;
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error (listMicrophoneNames init): " << Pa_GetErrorText(err) << std::endl;
        return names;
    }

    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 0) {
        std::cerr << "PortAudio error (listMicrophoneNames device count): " << Pa_GetErrorText(numDevices) << std::endl;
        Pa_Terminate();
        return names;
    }

    const PaDeviceInfo *deviceInfo;
    for (int i = 0; i < numDevices; ++i) {
        deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            names.push_back(deviceInfo->name);
        }
    }

    Pa_Terminate();
    return names;
}

PortAudioRecorder::PortAudioRecorder() 
    : stream(nullptr), recordingActive(false), energyThreshold(1000), 
      sampleRate_(SAMPLE_RATE), recordTimeout_(2.0), phraseTimeout_(3.0),
      max_buffer_samples_(0), max_silence_chunks_(0), consecutive_silence_chunks_(0),
      bypass_vad_(false) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error (init): " << Pa_GetErrorText(err) << std::endl;
        throw std::runtime_error("Failed to initialize PortAudio.");
    }
}

PortAudioRecorder::~PortAudioRecorder() {
    if (recordingActive) {
        stopRecording();
    }
    if (stream) {
        Pa_CloseStream(stream);
    }
    Pa_Terminate();
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

    // Bypass VAD during ambient noise calibration
    if (recorder->bypass_vad_) {
        std::vector<int16_t> chunk(in, in + framesPerBuffer);
        recorder->audioCallback(chunk);
        return paContinue;
    }

    std::vector<int16_t> current_chunk(in, in + framesPerBuffer);

    // Calculate RMS for VAD
    double sum_squares = 0.0;
    for (int16_t sample : current_chunk) {
        sum_squares += static_cast<double>(sample) * sample;
    }
    double rms = std::sqrt(sum_squares / framesPerBuffer);

    {
        std::lock_guard<std::mutex> lock(recorder->vad_buffer_mutex);
        if (rms > recorder->getEnergyThreshold()) {
            // Reset silence counter on voice activity
            if (recorder->consecutive_silence_chunks_ > 0) {
                recorder->consecutive_silence_chunks_ = 0;
            }
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
            recorder->audioCallback(recorder->vad_buffer);
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
    audioCallback = callback;
    sampleRate_ = sampleRate;
    recordTimeout_ = recordTimeout;
    phraseTimeout_ = phraseTimeout;
    
    // Calculate buffer limits
    max_buffer_samples_ = static_cast<size_t>(sampleRate_ * recordTimeout_);
    max_silence_chunks_ = static_cast<size_t>(std::ceil(phraseTimeout_ * sampleRate_ / FRAMES_PER_BUFFER));
    consecutive_silence_chunks_ = 0;
    bypass_vad_ = false;
    recordingActive = true;

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

void PortAudioRecorder::adjustForAmbientNoise() {
    std::cout << "Adjusting for ambient noise (listening for 3 seconds)..." << std::endl;
    std::vector<int16_t> noise_samples;
    std::mutex noise_mutex;
    std::condition_variable noise_cv;
    bool noise_collection_done = false;

    // Temporarily override the audio callback for noise collection
    auto original_callback = audioCallback;
    audioCallback = [&](const std::vector<int16_t>& audio_data) {
        std::lock_guard<std::mutex> lock(noise_mutex);
        noise_samples.insert(noise_samples.end(), audio_data.begin(), audio_data.end());
        if (noise_samples.size() >= static_cast<size_t>(SAMPLE_RATE * 3)) {
            noise_collection_done = true;
            noise_cv.notify_one();
        }
    };

    // Enable VAD bypass for raw audio collection
    bypass_vad_ = true;

    // Start the stream if not already running
    bool was_running = recordingActive;
    if (!was_running) {
        if (!startRecording(audioCallback, sampleRate_, recordTimeout_, phraseTimeout_)) {
            std::cerr << "Failed to start recording for ambient noise adjustment." << std::endl;
            audioCallback = original_callback;
            bypass_vad_ = false;
            return;
        }
    }

    // Wait for noise collection to complete
    std::unique_lock<std::mutex> lock(noise_mutex);
    auto status = noise_cv.wait_for(lock, std::chrono::seconds(4), [&]{ return noise_collection_done; });

    // Restore original settings
    bypass_vad_ = false;
    audioCallback = original_callback;
    
    if (!was_running) {
        stopRecording();
    }

    if (noise_samples.empty()) {
        std::cerr << "No noise samples collected for adjustment. Using default energy threshold." << std::endl;
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
    whisper_context *ctx;
    std::string model_path;

public:
    WhisperModel(const std::string& modelPath) : ctx(nullptr), model_path(modelPath) {
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
        if (!ctx) {
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

    void setLanguage(const std::string& lang) {
        // Actual language setting happens in transcribe()
    }
};

struct Args {
    std::string model = "medium";
    bool non_english = false;
    int energy_threshold = 1000;
    double record_timeout = 2.0;
    double phrase_timeout = 3.0;
    std::string language = "en";
    bool pipe = false;
    std::string default_microphone;
    std::string whisper_model_path;
};

Args parse_arguments(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--non_english") {
            args.non_english = true;
        } else if (arg == "--energy_threshold" && i + 1 < argc) {
            args.energy_threshold = std::stoi(argv[++i]);
        } else if (arg == "--record_timeout" && i + 1 < argc) {
            args.record_timeout = std::stod(argv[++i]);
        } else if (arg == "--phrase_timeout" && i + 1 < argc) {
            args.phrase_timeout = std::stod(argv[++i]);
        } else if (arg == "--language" && i + 1 < argc) {
            args.language = argv[++i];
        } else if (arg == "--pipe") {
            args.pipe = true;
        } else if (arg == "--default_microphone" && i + 1 < argc) {
            args.default_microphone = argv[++i];
        } else if (arg == "--whisper_model_path" && i + 1 < argc) {
            args.whisper_model_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model <name>            Model to use (tiny, base, small, medium, large). Default: medium\n"
                      << "  --non_english             Don't use the English-specific model variant.\n"
                      << "  --energy_threshold <int>  Energy level for mic to detect. Default: 1000\n"
                      << "  --record_timeout <float>  Max duration for audio chunks. Default: 2.0\n"
                      << "  --phrase_timeout <float>  Silence duration to end a phrase. Default: 3.0\n"
                      << "  --language <lang>         Language for transcription (de, en, es, fr, he, it, se). Default: en\n"
                      << "  --pipe                    Enable pipe mode for continuous streaming.\n"
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

// Trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
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
    } catch (const std::runtime_error& e) {
        std::cerr << "Failed to initialize PortAudio: " << e.what() << std::endl;
        return 1;
    }

#ifdef __linux__
    if (!args.default_microphone.empty() && args.default_microphone != "list") {
        std::cerr << "Warning: --default_microphone is specified, but PortAudioRecorder currently uses the system default." << std::endl;
    }
#endif

    recorder->setEnergyThreshold(args.energy_threshold);

    WhisperModel audio_model(args.whisper_model_path);
    std::vector<std::string> transcription = {""};

    // Calibrate microphone
    std::cout << "Calibrating microphone..." << std::endl;
    if (!recorder->startRecording(
            [](const std::vector<int16_t>&) {}, // Dummy callback
            SAMPLE_RATE, args.record_timeout, args.phrase_timeout)) {
        std::cerr << "Failed to start recording for calibration." << std::endl;
        return 1;
    }
    recorder->adjustForAmbientNoise();
    recorder->stopRecording();

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
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::milliseconds(250), [&]{ return !data_queue.empty(); });

            auto now = std::chrono::system_clock::now();
            bool phrase_complete = false;

            if (phrase_time_set && (now - last_phrase_end_time) > std::chrono::duration<double>(args.phrase_timeout)) {
                phrase_complete = true;
            }

            if (!data_queue.empty()) {
                std::vector<int16_t> collected_audio_data;
                while (!data_queue.empty()) {
                    const auto& chunk = data_queue.front();
                    collected_audio_data.insert(collected_audio_data.end(), chunk.begin(), chunk.end());
                    data_queue.pop();
                }
                last_phrase_end_time = now;
                phrase_time_set = true;

                // Pad audio to minimum length if needed
                if (collected_audio_data.size() < MIN_AUDIO_SAMPLES) {
                    collected_audio_data.resize(MIN_AUDIO_SAMPLES, 0);
                }

                // Convert to float and normalize
                std::vector<float> audio_np(collected_audio_data.size());
                for (size_t i = 0; i < collected_audio_data.size(); ++i) {
                    audio_np[i] = static_cast<float>(collected_audio_data[i]) / 32768.0f;
                }

                std::string text = audio_model.transcribe(audio_np, args.language);
                text = trim(text);

                // Skip empty transcriptions
                if (text.empty()) {
                    continue;
                }

                if (args.pipe) {
                    std::cout << text << std::endl;
                } else {
                    if (phrase_complete) {
                        transcription.push_back(text);
                    } else {
                        transcription.back() = text;
                    }

                    clear_console();
                    for (const auto& line : transcription) {
                        std::cout << line << std::endl;
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
