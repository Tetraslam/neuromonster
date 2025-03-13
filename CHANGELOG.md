# Changelog

All notable changes to the NeuroMonster project will be documented in this file.

## [Unreleased]

### Added
- Initial project structure with core components
- Basic README with installation instructions and project overview

### Changed
- Improved PyTorch dependency management using `uv` optional dependencies
- Added support for both CPU and CUDA 12.6 versions of PyTorch
- Updated installation instructions in README.md

### Fixed
- Fixed PyTorch installation issues by implementing proper optional dependencies
- Ensured that `uv pip install` uses the correct syntax for optional dependencies
- Corrected installation commands to use `uv pip install -r pyproject.toml --extra` instead of `uv pip sync --extra`

## How to Use PyTorch with NeuroMonster

### CPU Version
To use the CPU-only version of PyTorch:
```bash
uv pip install -r pyproject.toml --extra cpu
```

### CUDA 12.6 Version
To use the CUDA 12.6 version of PyTorch (requires NVIDIA GPU with CUDA 12.6 installed):
```bash
uv pip install -r pyproject.toml --extra cu126
```

The configuration in `pyproject.toml` ensures that:
1. The correct PyTorch packages are installed from the appropriate index
2. Users can't accidentally install both CPU and CUDA versions simultaneously
3. Running `uv pip install` later won't remove the installed PyTorch packages 