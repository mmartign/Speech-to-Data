# Contributing to Speech-to-Knowledge

Thanks for your interest in contributing! This project is a real-time
medical speech-to-knowledge pipeline written in C++20 and built with CMake.

## Code of Conduct

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold it.

## Getting Started

1. Fork the repository and clone your fork.
2. Follow the build instructions in [README.md](README.md) to configure and
   build the project with CMake.
3. Create a branch for your change: `git checkout -b my-change`.

## Building and Testing

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The `analyze_text_tests` and `transcribe_audio_tests` targets build and run
without the macOS-only audio/whisper.cpp dependencies, so they're the
fastest way to validate pure-logic changes; they're also what CI runs on
every push and pull request (see `.github/workflows/ci.yml`).

## Making Changes

* Keep changes focused and scoped to a single concern.
* Match the existing code style in the file you're editing.
* Add or update unit tests under `tests/` for any behavior change.
* Ensure `ctest` passes before opening a pull request.
* Write clear commit messages that explain *why* a change was made, not just
  what changed.

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a pull request against `main`, describing the motivation and the
   testing you performed.
3. Link any related issues.
4. Be responsive to review feedback.

## Reporting Bugs and Requesting Features

Please use the issue templates when opening a new issue — they help us get
the information needed to triage quickly.

## Security Issues

Do not open a public issue for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for how to report them responsibly.

## License

By contributing, you agree that your contributions will be licensed under
the project's [AGPL-3.0-or-later](LICENSE) license.
