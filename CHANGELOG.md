# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Package dependencies are now listed in pyproject.toml
- Initialized CHANGELOG.md to track changes
- Ignore local .vscode settings in .gitignore
- Ignore macOS-specific .DS_Store files in .gitignore
- Added installation and testing instructions to README.md
- Added citation and zenodo references to README.md
- The folder [examples](examples/) now contains usage examples
- [examples/getting-started.ipynb](examples/getting-started.ipynb) demonstrates the basic problem instance creation workflow
- Python version pinned to 3.13 (.python-version)
- Added uv.lock for better requirements management

### Changed

- pyproject.toml requires python>=3.11 in line with numpy>=2.4.0 requirement

### Fixed

- Dominance plots now correctly compute log(layer + 1) instead of log(layer) to avoid log(0) for nondominated points.

## [0.0.1] - 2026-01-09

### Added

- Initial migration and split of package code from experimental code for "BONO-Bench: A Comprehensive Test Suite for Bi-objective Numerical Optimization with Traceable Pareto Sets" paper

[Unreleased]: https://github.com/schaepermeier/bonobench/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/schaepermeier/bonobench/releases/tag/v0.0.1
