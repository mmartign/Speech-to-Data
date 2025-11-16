all: analyze_text.exe  transcribe_audio.exe transcribe_audio_adaptive_energy.exe

analyze_text.exe: analyze_text.cpp
	g++ -std=c++20 -I ../openai-cpp/include/openai -o analyze_text.exe analyze_text.cpp -lcurl

transcribe_audio.exe: transcribe_audio.cpp
	g++ -std=c++20 -o transcribe_audio.exe transcribe_audio.cpp -I /opt/local/include -I ../whisper.cpp/include -I ../whisper.cpp/ggml/include /opt/local/lib/libportaudio.dylib ../whisper.cpp/build/src/libwhisper.dylib -rpath /usr/local/lib

transcribe_audio_adaptive_energy.exe: transcribe_audio_adaptive_energy.cpp
	g++ -std=c++20 -o transcribe_audio_adaptive_energy.exe transcribe_audio_adaptive_energy.cpp -I /opt/local/include -I ../whisper.cpp/include -I ../whisper.cpp/ggml/include /opt/local/lib/libportaudio.dylib ../whisper.cpp/build/src/libwhisper.dylib -rpath /usr/local/lib
