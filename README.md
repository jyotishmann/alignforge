# AlignForge

> LLM post-training pipeline: QLoRA SFT → DPO alignment → judge-based evaluation → GGUF serving.

**Under construction.**
[![CI](https://github.com/<your-username>/alignforge/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/alignforge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Quick start

```bash
make setup       # creates venv and installs CPU deps
make check       # lint + typecheck + test + import contracts
alignforge doctor
