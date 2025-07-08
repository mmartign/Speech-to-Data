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

// PortAudio includes
#include "portaudio.h"

// whisper.cpp includes
// You'll need to adjust this path based on where you put whisper.cpp
#include "whisper.h"

// --- Placeholder for a C++ Audio Capture Library (PortAudio Implementation) ---
#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 1024 // A common buffer size for audio processing

class AudioRecorder {
public:
    virtual ~AudioRecorder() = default;
    virtual bool startRecording(std::function<void(const std::vector<int16_t>&)> callback, int sampleRate, double recordTimeout) = 0;
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

    // Buffer to accumulate audio before VAD processing
    std::vector<int16_t> vad_buffer;
    std::mutex vad_buffer_mutex; // Protects vad_buffer

    static int pa_callback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData);

public:
    PortAudioRecorder();
    ~PortAudioRecorder();

    bool startRecording(std::function<void(const std::vector<int16_t>&)> callback, int sampleRate, double recordTimeout) override;
    void stopRecording() override;
    void adjustForAmbientNoise() override;
    void setEnergyThreshold(int threshold) override { energyThreshold = threshold; }
    int getEnergyThreshold() const override { return energyThreshold; }
};

// Static method to list microphone names using PortAudio
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

PortAudioRecorder::PortAudioRecorder() : stream(nullptr), recordingActive(false), energyThreshold(1000), sampleRate_(SAMPLE_RATE), recordTimeout_(2.0) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error (init): " << Pa_GetErrorText(err) << std::endl;
        throw std::runtime_error("Failed to initialize PortAudio.");
    }
}

PortAudioRecorder::~PortAudioRecorder() {
    if (stream) {
        Pa_StopStream(stream);
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
        // No input data, might be a problem or end of stream
        return paContinue;
    }

    std::vector<int16_t> current_chunk(in, in + framesPerBuffer);

    // Basic Voice Activity Detection (VAD) based on energy threshold
    double sum_squares = 0.0;
    for (int16_t sample : current_chunk) {
        sum_squares += static_cast<double>(sample) * sample;
    }
    // RMS (Root Mean Square)
    double rms = std::sqrt(sum_squares / framesPerBuffer);

    if (rms > recorder->getEnergyThreshold()) {
        std::lock_guard<std::mutex> lock(recorder->vad_buffer_mutex);
        recorder->vad_buffer.insert(recorder->vad_buffer.end(), current_chunk.begin(), current_chunk.end());

        // If enough data has accumulated (e.g., 16000 samples for 1 second at 16kHz)
        // or a certain "record_timeout" worth of data, send it.
        // This logic is a simplification of the original `record_timeout`.
        // The original script put everything into the queue; here we're more selective.
        if (recorder->vad_buffer.size() >= recorder->sampleRate_ * recorder->recordTimeout_) {
             recorder->audioCallback(recorder->vad_buffer);
             recorder->vad_buffer.clear();
        }
    } else {
        // If energy is below threshold, clear any pending VAD buffer
        // This simulates `phrase_timeout` by not accumulating silence
        std::lock_guard<std::mutex> lock(recorder->vad_buffer_mutex);
        if (!recorder->vad_buffer.empty()) {
            // Send whatever was collected if it's less than record_timeout but still had voice
            if (rms < recorder->getEnergyThreshold() * 0.5) { // A bit of hysteresis
                 recorder->audioCallback(recorder->vad_buffer);
                 recorder->vad_buffer.clear();
            }
        }
    }

    return paContinue;
}

bool PortAudioRecorder::startRecording(std::function<void(const std::vector<int16_t>&)> callback, int sampleRate, double recordTimeout) {
    audioCallback = callback;
    sampleRate_ = sampleRate;
    recordTimeout_ = recordTimeout;
    recordingActive = true;

    PaStreamParameters inputParameters;
    inputParameters.device = Pa_GetDefaultInputDevice(); // default input device
    if (inputParameters.device == paNoDevice) {
        std::cerr << "Error: No default input device." << std::endl;
        return false;
    }
    inputParameters.channelCount = 1;                   // mono input
    inputParameters.sampleFormat = paInt16;             // 16-bit int
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(
        &stream,
        &inputParameters,
        nullptr, // No output
        sampleRate_,
        FRAMES_PER_BUFFER, // framesPerBuffer
        paClipOff, // we won't output out of range samples so don't bother clipping them
        pa_callback,
        this // Pass 'this' as userData to the callback
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
    }
}

void PortAudioRecorder::adjustForAmbientNoise() {
    std::cout << "Adjusting for ambient noise (listening for 3 seconds)..." << std::endl;
    // Collect some samples to estimate noise level
    std::vector<int16_t> noise_samples;
    std::mutex noise_mutex;
    std::condition_variable noise_cv;
    bool noise_collection_done = false;

    // Temporarily override the main audio callback for noise collection
    auto original_callback = audioCallback;
    audioCallback = [&](const std::vector<int16_t>& audio_data) {
        std::lock_guard<std::mutex> lock(noise_mutex);
        noise_samples.insert(noise_samples.end(), audio_data.begin(), audio_data.end());
        if (noise_samples.size() >= SAMPLE_RATE * 3) { // Collect 3 seconds of noise
            noise_collection_done = true;
            noise_cv.notify_one();
        }
    };

    // Make sure the stream is active for this to work
    if (!stream) {
        std::cerr << "Stream not active for ambient noise adjustment. Please start recording first." << std::endl;
        audioCallback = original_callback; // Restore immediately if error
        return;
    }

    // Wait for noise collection to complete
    std::unique_lock<std::mutex> lock(noise_mutex);
    noise_cv.wait_for(lock, std::chrono::seconds(3), [&]{ return noise_collection_done; });

    // Restore original callback
    audioCallback = original_callback;

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

    // Set threshold based on a multiplier of the estimated noise RMS
    // This multiplier can be tuned. 2.0-3.0 is often a good starting point.
    energyThreshold = static_cast<int>(rms * 2.5);
    std::cout << "Adjusted energy threshold to: " << energyThreshold << std::endl;
}

// --- Integration with whisper.cpp ---
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
            return "Error: Whisper model not loaded.";
        }

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        // Set language
        params.language = lang.c_str();
        params.n_threads = std::min(4, (int)std::thread::hardware_concurrency()); // Use up to 4 threads or available cores

        // Run transcription
        if (whisper_full(ctx, params, audio_data_normalized.data(), audio_data_normalized.size()) != 0) {
            return "Error: Failed to transcribe audio.";
        }

        std::string result_text = "";
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char *text = whisper_full_get_segment_text(ctx, i);
            result_text += text;
        }
        return result_text;
    }

    // For whisper.cpp, language is set per transcription call, not on the model object.
    void setLanguage(const std::string& lang) {
        // This function exists for API compatibility with the Python code,
        // but the actual language setting happens in transcribe().
    }
};

// --- Command Line Argument Parsing (simplified) ---
struct Args {
    std::string model = "medium";
    bool non_english = false;
    int energy_threshold = 10;
    double record_timeout = 2.0;
    double phrase_timeout = 3.0;
    std::string language = "en";
    bool pipe = false;
    std::string default_microphone = ""; // For Linux, if implemented
    std::string whisper_model_path = ""; // Path to the ggml model file
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
                      << "  --record_timeout <float>  Time window for real-time audio chunking. Default: 2.0\n"
                      << "  --phrase_timeout <float>  Pause duration before starting a new transcription line. Default: 3.0\n"
                      << "  --language <lang>         Language for transcription (de, en, es, fr, he, it, se). Default: en\n"
                      << "  --pipe                    Enable pipe mode for continuous streaming.\n"
                      << "  --whisper_model_path <path> REQUIRED: Path to the ggml Whisper model file (e.g., ggml-medium.bin)\n";
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

// Function to clear console (platform-specific)
void clear_console() {
    #ifdef _WIN32
        system("cls");
    #else
        // Use ANSI escape code for clearing screen and moving cursor to top-left
        std::cout << "\033[2J\033[H";
    #endif
}

int main(int argc, char* argv[]) {
    // 1. Argument parsing
    Args args = parse_arguments(argc, argv);

    // 2. Variables
    std::chrono::time_point<std::chrono::system_clock> last_phrase_end_time;
    bool phrase_time_set = false;

    // Thread-safe queue for audio data
    // The PortAudio callback will push full chunks based on its internal VAD/buffering.
    std::queue<std::vector<int16_t>> data_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    // 3. Microphone setup
    std::unique_ptr<PortAudioRecorder> recorder;
    try {
        recorder = std::make_unique<PortAudioRecorder>();
    } catch (const std::runtime_error& e) {
        std::cerr << "Failed to initialize PortAudio: " << e.what() << std::endl;
        return 1;
    }

#ifdef __linux__
    if (!args.default_microphone.empty()) {
        if (args.default_microphone == "list") {
            std::cout << "Available microphone devices:" << std::endl;
            for (const auto& name : AudioRecorder::listMicrophoneNames()) {
                std::cout << "- " << name << std::endl;
            }
            return 0;
        }
        // PortAudioRecorder currently uses default device.
        // To select by name, you'd need to modify PortAudioRecorder to
        // accept a device index, and then loop through Pa_GetDeviceInfo
        // to find the matching index for the name.
        std::cerr << "Warning: --default_microphone is specified, but PortAudioRecorder currently uses the system default. "
                  << "Modify PortAudioRecorder to select a specific device by name if needed." << std::endl;
    }
#endif

    recorder->setEnergyThreshold(args.energy_threshold);

    // 4. Load Whisper model
    WhisperModel audio_model(args.whisper_model_path);

    std::vector<std::string> transcription = {""};

    // 5. Calibrate microphone
    // Start recording temporarily for calibration
    if (!recorder->startRecording(
            [&](const std::vector<int16_t>& audio_data) {
                // This callback for calibration will push data to a temp queue or buffer.
                // The adjustForAmbientNoise() method will handle this internally.
            },
            SAMPLE_RATE, args.record_timeout)) {
        std::cerr << "Failed to start recording for calibration." << std::endl;
        return 1;
    }
    recorder->adjustForAmbientNoise();
    recorder->stopRecording(); // Stop calibration recording


    // 6. Start background listening for transcription
    auto record_callback = [&](const std::vector<int16_t>& audio_data) {
        // This is the actual callback for continuous transcription
        std::unique_lock<std::mutex> lock(queue_mutex);
        data_queue.push(audio_data);
        queue_cv.notify_one();
    };

    if (!recorder->startRecording(record_callback, SAMPLE_RATE, args.record_timeout)) {
        std::cerr << "Failed to start continuous recording." << std::endl;
        return 1;
    }

    if (!args.pipe) {
        std::cout << "Model loaded and recording started.\n" << std::endl;
    }

    // 7. Main loop
    try {
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // Wait until there's data or a short timeout
            queue_cv.wait_for(lock, std::chrono::milliseconds(250), [&]{ return !data_queue.empty(); });

            auto now = std::chrono::system_clock::now();
            bool phrase_complete = false;

            // Check if enough time has passed since the last detected phrase
            if (phrase_time_set && (now - last_phrase_end_time) > std::chrono::duration<double>(args.phrase_timeout)) {
                phrase_complete = true;
            }

            if (!data_queue.empty()) {
                // Process audio buffer
                std::vector<int16_t> collected_audio_data;
                while (!data_queue.empty()) {
                    const auto& chunk = data_queue.front();
                    collected_audio_data.insert(collected_audio_data.end(), chunk.begin(), chunk.end());
                    data_queue.pop();
                }
                last_phrase_end_time = now; // Update time of last activity
                phrase_time_set = true;

                // Convert int16_t to float32 and normalize
                std::vector<float> audio_np(collected_audio_data.size());
                for (size_t i = 0; i < collected_audio_data.size(); ++i) {
                    audio_np[i] = static_cast<float>(collected_audio_data[i]) / 32768.0f;
                }

                std::string text = audio_model.transcribe(audio_np, args.language);
                // Basic trim of leading/trailing whitespace
                size_t first = text.find_first_not_of(" \t\n\r\f\v");
                if (std::string::npos != first) {
                    size_t last = text.find_last_not_of(" \t\n\r\f\v");
                    text = text.substr(first, (last - first + 1));
                } else {
                    text = ""; // String was all whitespace
                }


                if (args.pipe) {
                    std::cout << text << std::endl;
                    std::flush(std::cout);
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
                // If no audio data for a while, and a phrase has been completed,
                // consider the last phrase truly complete if no new sound detected.
                if (phrase_time_set && (now - last_phrase_end_time) > std::chrono::duration<double>(args.phrase_timeout * 1.5)) {
                    // Force a new line if significant silence.
                    // This mirrors the Python logic where `phrase_complete` is true
                    // if no data arrives for `phrase_timeout`.
                    if (!transcription.back().empty()) { // Only add new line if last was not empty
                        transcription.push_back("");
                    }
                    phrase_time_set = false; // Reset to allow next phrase to start
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }

    // Clean up
    recorder->stopRecording();

    if (!args.pipe) {
        std::cout << "\n\nTranscription:" << std::endl;
        for (const auto& line : transcription) {
            std::cout << line << std::endl;
        }
    }

    return 0;
}