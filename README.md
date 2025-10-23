# Gemini Migration

A toolkit for converting API requests between different LLM formats (Claude, OpenAI, Gemini) and testing them against Gemini API endpoints.

## Overview

This project provides converters and tools to migrate API requests from Claude and OpenAI formats to Gemini format, and to call Gemini APIs with various configurations.

## Components

### Converters

#### `converter_claude2gemini.py`
Converts Claude API requests to Gemini format.

**Usage:**
```bash
python3 converter_claude2gemini.py [--input_folder INPUT] [--output_folder OUTPUT]
```

**Parameters:**
- `--input_folder`: Input folder containing Claude request JSON files (default: `claude_requests`)
- `--output_folder`: Output folder for converted files (default: `<input_folder>_to_gemini`)

**Features:**
- Converts message roles (user→user, assistant→model, system→systemInstruction)
- Converts tool calls from Claude's `tool_use` to Gemini's `functionCall`
- Converts tool responses from `tool_result` to `functionResponse`
- Converts tools array to `functionDeclarations` format
- Handles multi-modal content (text + images)
- Processes files recursively in folders and subfolders

---

#### `converter_claude2openai.py`
Converts Claude API requests to OpenAI format.

**Usage:**
```bash
python3 converter_claude2openai.py [--input_folder INPUT] [--output_folder OUTPUT]
```

**Parameters:**
- `--input_folder`: Input folder containing Claude request JSON files (default: `claude_requests`)
- `--output_folder`: Output folder for converted files (default: `<input_folder>_to_openai`)

**Features:**
- Converts message roles (maintains same roles as Claude)
- Converts tool calls from Claude's format to OpenAI's `tool_calls` format
- Converts tool responses to OpenAI's tool message format
- Converts tools array to OpenAI's function format
- Processes files recursively in folders and subfolders

---

#### `converter_openai2gemini.py`
Converts OpenAI API requests to Gemini format.

**Usage:**
```bash
python3 converter_openai2gemini.py [--input_folder INPUT] [--output_folder OUTPUT]
```

**Parameters:**
- `--input_folder`: Input folder containing OpenAI request JSON files (default: `openai_requests`)
- `--output_folder`: Output folder for converted files (default: `<input_folder>_to_gemini`)

**Features:**
- Converts message roles (user→user, assistant→model, system→systemInstruction, tool→user with functionResponse)
- Converts tool calls from OpenAI's format to Gemini's `functionCall`
- Extracts tool names from tool_call_id mapping when not present in tool messages
- Converts tools array to `functionDeclarations` format
- Handles multi-modal content including base64-encoded images
- Converts enum values to strings as required by Gemini
- Processes files recursively in folders and subfolders

---

### API Caller

#### `gemini_api_caller.py`
Calls Gemini API with converted requests and supports both Gemini native and OpenAI-compatible endpoints.

**Usage:**
```bash
python3 gemini_api_caller.py [OPTIONS]
```

**Parameters:**
- `--input-folder`: Input folder containing Gemini request JSON files (default: `claude_requests_to_gemini`)
- `--output-folder`: Output folder for results (default: `claude_requests_to_gemini_output`)
- `--iterations`: Number of iterations to process all requests (default: `1`)
- `--function-call-mode`: Function calling mode - `auto`, `any`, or `validated` (default: `auto`)
- `--fc2`: Use cloud-llm-preview4 project (default: `False`)
- `--thinking-budget`: Thinking budget value. If 0, generationConfig is not added (default: `0`)
- `--project`: GCP project ID (default: `cloud-llm-preview4`)
- `--model-name`: Model name to use (default: `gemini-2.5-pro`)
- `--openai-endpoint`: Use OpenAI-compatible endpoint (default: `False`)
- `--location`: GCP location/region (default: `global`)

**Features:**
- Supports both Gemini native and OpenAI-compatible endpoints
- Handles regional endpoints (e.g., `us-central1-aiplatform.googleapis.com`)
- Adds model field for OpenAI endpoint requests
- Processes responses in both Gemini and OpenAI formats
- Supports multiple function calling modes
- Adds labels with timestamp, session ID, and configuration settings
- Processes files folder by folder in alphabetical order
- Handles errors gracefully with detailed error reporting
- Supports multiple iterations for testing

**URL Construction:**
- **Gemini endpoint**: `https://[location-]aiplatform.googleapis.com/[v1|v1beta1]/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
- **OpenAI endpoint**: `https://[location-]aiplatform.googleapis.com/[v1|v1beta1]/projects/{project}/locations/{location}/endpoints/openapi/chat/completions`

---

### Testing Script

#### `run_e2e_eval.sh`
Runs end-to-end evaluation with different Gemini API configurations.

**Usage:**
```bash
./run_e2e_eval.sh <iterations>
```

**Parameters:**
- `iterations`: Number of iterations to run for each configuration

**Configurations:**
1. **FC1 mode**: `fc2=false`, `function-call-mode=auto`
2. **FC2 mode**: `fc2=true`, `function-call-mode=auto`
3. **FC2 validated mode**: `fc2=true`, `function-call-mode=validated`

**Output:**
- `claude_requests_to_gemini_output_FC1/`
- `claude_requests_to_gemini_output_FC2/`
- `claude_requests_to_gemini_output_FC2_validate/`

---

### Analysis Tools

#### `compare_function_calls.py`
*Ongoing*

#### `read_google_sheet.py`
*Ongoing*

---

## Workflow Example

### 1. Convert Claude requests to Gemini format
```bash
python3 converter_claude2gemini.py --input_folder claude_requests --output_folder claude_requests_to_gemini
```

### 2. Convert Claude requests to OpenAI format
```bash
python3 converter_claude2openai.py --input_folder claude_requests --output_folder claude_requests_to_openai
```

### 3. Convert OpenAI requests to Gemini format
```bash
python3 converter_openai2gemini.py --input_folder openai_requests --output_folder openai_requests_to_gemini
```

### 4. Call Gemini API with converted requests

**Using Gemini native endpoint:**
```bash
python3 gemini_api_caller.py \
  --input-folder claude_requests_to_gemini \
  --output-folder claude_requests_to_gemini_output \
  --fc2 true \
  --project cloud-llm-preview1
```

**Using OpenAI-compatible endpoint:**
```bash
python3 gemini_api_caller.py \
  --input-folder claude_requests_to_openai \
  --output-folder claude_requests_to_openai_output \
  --fc2 true \
  --project cloud-llm-preview1 \
  --openai-endpoint true
```

### 5. Run end-to-end evaluation
```bash
./run_e2e_eval.sh 3
```

---

## Input Folder Structure

The converters and API caller support recursive folder processing:

```
claude_requests/
├── request_1.json
├── request_2.json
├── subfolder_1/
│   ├── request_3.json
│   └── request_4.json
└── subfolder_2/
    ├── request_5.json
    └── request_6.json
```

All JSON files in the parent folder and subfolders will be processed while maintaining the folder structure in the output.

---

## Requirements

- Python 3.x
- `requests` library
- `google-auth` library
- Google Cloud credentials with Vertex AI API access

---

## Authentication

The tools use Google Cloud Application Default Credentials. Ensure you are authenticated:

```bash
gcloud auth application-default login
```

---

## Notes

- All converters preserve folder structure in output
- The API caller processes files folder by folder in alphabetical order
- Results include timestamps, session IDs, and configuration labels
- OpenAI endpoint support automatically handles format differences in requests and responses
