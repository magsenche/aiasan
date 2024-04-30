# AIASAN

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

AIASAN (Artificial Intelligence Agent SANdbox) is an open-source toolbox designed to provide a foundation for building AI agents, Large Language Model Tools and other artificial intelligence applications.

This sandbox aims to simplify the development process by providing pre-built functions, utilities, and tools that can be easily integrated into your projects.

## Cookbook
A collection of example scripts demonstrating how to use the toolbox functions and utilities:

### Askzono
An innovative local AI-powered chat application that enables users to engage with their documents.

- run `pdm askzono` or `streamlit run cookbook/askzono.py`
- integrate your documents (markdown or pdf)
- converse with your documents via the chat



https://github.com/magsenche/aiasan/assets/102949971/998403d0-8e0e-4ca6-b4cd-dec6bcf11c2a



## Library

1. **Tools**: A ready-to-use toolbox for AI agents.
2. **Prompts**: A prompt hub for AI agents.
3. **VectorStore**: A FAISS-based **updatable** vector store for indexing and retrieving documents.

## Examples

- [ai assistant](https://github.com/magsenche/hanazono/blob/main/examples/aiassistant.py)

## Getting Started

1. clone this repository using Git: `git clone https://github.com/magsenche/aiasan.git`
2. install required dependencies by running: `pdm install`
3. setup your [environment variables](#environment-variable)
4. explore the `cookbook` folder for example scripts showcasing toolbox usage
5. start building your AI agent or LLM tool using the provided utilities!

### Environment variable

```ini
LOCAL_MODEL = "llama3:latest"
LOCAL_EMBED_MODEL = "nomic-embed-text:latest"
OUTPUT_FOLDER = "outputs"
```

## TODO
### Features
- [ ] askzono: enable loading an existing vectorstore

### Documentation
- [x] askzono: video demo
