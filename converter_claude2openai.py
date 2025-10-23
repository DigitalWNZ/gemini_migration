#!/usr/bin/env python3
"""
Claude to OpenAI API Request Converter

This script converts API requests from Claude's format to OpenAI's format.

1. Message Conversion

- Roles mapping:
- user → user (same)
- assistant → assistant (same)
- system → system (same)
- tool → tool (same)
- Content format:
- Simple text: stays as string
- Multi-modal content: converted from Claude's array format to OpenAI's array format

2. Tool Calls (from assistant)

- Claude format:
{
"role": "assistant",
"content": [
    {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
]
}
- OpenAI format:
{
"role": "assistant",
"content": null,
"tool_calls": [{
    "id": "...",
    "type": "function",
    "function": {
    "name": "...",
    "arguments": "..." (JSON string)
    }
}]
}

3. Tool Results/Responses

- Claude format:
{
"role": "user",
"content": [
    {"type": "tool_result", "tool_use_id": "...", "content": "..."}
]
}
- OpenAI format:
{
"role": "tool",
"tool_call_id": "...",
"name": "function_name",
"content": "..."
}
- Note: The function name is retrieved from the matching tool_use message with the same id

4. Tools/Functions Definition

- Claude format:
{
"name": "...",
"description": "...",
"input_schema": {...}
}
- OpenAI format:
{
"type": "function",
"function": {
    "name": "...",
    "description": "...",
    "parameters": {...}
}
}
"""

import argparse
import json
import os
import glob
import copy
from typing import Dict, List, Any, Optional


class ClaudeToOpenAIConverter:
    """Converts Claude API requests to OpenAI API format."""

    # Model mapping from Claude to OpenAI
    MODEL_MAPPING = {
        "claude-3-opus-20240229": "gpt-4-turbo-preview",
        "claude-3-sonnet-20240229": "gpt-4-turbo-preview",
        "claude-3-haiku-20240307": "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022": "gpt-4-turbo-preview",
        "claude-3-5-haiku-20241022": "gpt-3.5-turbo",
        # Add more mappings as needed
    }

    def __init__(self):
        """Initialize the converter."""
        pass

    def convert_messages(self, claude_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Claude message format to OpenAI format.

        Claude format: {"role": "user/assistant/system", "content": "text"}
        OpenAI format: {"role": "user/assistant/system", "content": "text"}
        """
        openai_messages = []

        # Build a mapping of tool_use_id to function name from assistant messages
        tool_use_map = {}
        for message in claude_messages:
            if message.get("role") == "assistant" and isinstance(message.get("content"), list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_use_id = item.get("id")
                        function_name = item.get("name")
                        if tool_use_id and function_name:
                            tool_use_map[tool_use_id] = function_name

        for message in claude_messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # OpenAI uses the same role names as Claude
            openai_role = role

            # Convert content to OpenAI format
            if isinstance(content, str):
                openai_content = content
            elif isinstance(content, list):
                # Handle multi-modal content
                openai_content_parts = []
                for item in content:
                    if isinstance(item, str):
                        openai_content_parts.append({
                            "type": "text",
                            "text": item
                        })
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            openai_content_parts.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") == "image":
                            # Convert image format
                            source = item.get("source", {})
                            if source.get("type") == "base64":
                                openai_content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                                    }
                                })
                            elif source.get("type") == "url":
                                openai_content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": source.get("url", "")
                                    }
                                })
                        elif item.get("type") == "tool_use":
                            # Handle function calls - convert to OpenAI function call format
                            # This will be added to the assistant message in a special way
                            pass
                        elif item.get("type") == "tool_result":
                            # Handle tool results - convert to function response
                            # This becomes a tool message in OpenAI
                            pass

                if openai_content_parts:
                    openai_content = openai_content_parts
                else:
                    openai_content = str(content)
            else:
                openai_content = str(content)

            # Handle tool calls and tool results specially
            if isinstance(content, list):
                # Check for tool_use (function calls)
                tool_uses = [item for item in content if isinstance(item, dict) and item.get("type") == "tool_use"]
                if tool_uses and role == "assistant":
                    # Create OpenAI function call format
                    text_content = " ".join([
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ])

                    for tool_use in tool_uses:
                        openai_messages.append({
                            "role": "assistant",
                            "content": text_content if text_content else None,
                            "tool_calls": [{
                                "id": tool_use.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tool_use.get("name", ""),
                                    "arguments": json.dumps(tool_use.get("input", {}))
                                }
                            }]
                        })
                    continue

                # Check for tool_result
                tool_results = [item for item in content if isinstance(item, dict) and item.get("type") == "tool_result"]
                if tool_results:
                    for tool_result in tool_results:
                        content_str = tool_result.get("content", "")
                        # Get function name from the tool_use_map using tool_use_id
                        tool_use_id = tool_result.get("tool_use_id", "")
                        tool_name = tool_use_map.get(tool_use_id, "unknown")

                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "name": tool_name,
                            "content": content_str
                        })
                    continue

            openai_messages.append({
                "role": openai_role,
                "content": openai_content
            })

        return openai_messages

    def convert_tools(self, claude_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Claude tools (functions) to OpenAI function declarations.

        Claude format: tools with name, description, input_schema
        OpenAI format: tools with type, function (name, description, parameters)
        """
        if not claude_tools:
            return []

        openai_tools = []
        for tool in claude_tools:
            # Convert input_schema to parameters format
            input_schema = tool.get("input_schema", {})

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": input_schema  # OpenAI uses the same JSON Schema format
                }
            }

            openai_tools.append(openai_tool)

        return openai_tools

    def fix_request_issues(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix known issues in the request data during conversion."""
        # Deep copy to avoid modifying the original
        fixed_data = copy.deepcopy(request_data)

        # Check if tools exist in the request
        if 'tools' in fixed_data and isinstance(fixed_data['tools'], list):
            for tool in fixed_data['tools']:
                if 'function' in tool and isinstance(tool['function'], dict):
                    func = tool['function']
                    # Fix parameter mismatches
                    if func.get('name') == 'segment_anything':
                        if 'parameters' in func and 'required' in func['parameters'] and 'object' in func['parameters']['required']:
                            # Replace 'object' with 'object_english_name' in required array
                            func['parameters']['required'] = [
                                'object_english_name' if param == 'object' else param
                                for param in func['parameters']['required']
                            ]

                    elif func.get('name') == 'Pira_image2image':
                        if 'parameters' in func and 'required' in func['parameters'] and 'cfg' in func['parameters']['required']:
                            # Remove 'cfg' from required array as it doesn't exist in properties
                            func['parameters']['required'] = [
                                param for param in func['parameters']['required']
                                if param != 'cfg'
                            ]

                    elif func.get('name') == 'gemini_edit':
                        # Fix required parameter name
                        if 'parameters' in func and 'required' in func['parameters'] and 'image' in func['parameters']['required']:
                            # Replace 'image' with 'images' in required array
                            func['parameters']['required'] = [
                                'images' if param == 'image' else param
                                for param in func['parameters']['required']
                            ]

                        # Fix the type of images field from ["array", "null"] to "array"
                        if 'parameters' in func and 'properties' in func['parameters'] and 'images' in func['parameters']['properties']:
                            if isinstance(func['parameters']['properties']['images'].get('type'), list):
                                func['parameters']['properties']['images']['type'] = 'array'

                    elif func.get('name') == 'outpaint':
                        if 'parameters' in func and 'required' in func['parameters'] and 'prompt' in func['parameters']['required']:
                            # Replace 'prompt' with 'english_prompt' in required array
                            func['parameters']['required'] = [
                                'english_prompt' if param == 'prompt' else param
                                for param in func['parameters']['required']
                            ]

        return fixed_data

    def convert_request(self, claude_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a complete Claude API request to OpenAI format.

        Args:
            claude_request: Claude API request dictionary

        Returns:
            OpenAI API request dictionary
        """
        # Extract required fields
        messages = claude_request.get("messages", [])
        tools = claude_request.get("tools", [])

        # Convert messages
        openai_messages = self.convert_messages(messages)

        # Build OpenAI request
        openai_request = {
            "messages": openai_messages
        }

        # Convert and add tools if present
        if tools:
            openai_tools = self.convert_tools(tools)
            if openai_tools:
                openai_request["tools"] = openai_tools

        # Apply fixes to the converted request
        openai_request = self.fix_request_issues(openai_request)

        return openai_request

    def convert_from_json(self, claude_json: str) -> str:
        """
        Convert Claude request from JSON string to OpenAI JSON string.

        Args:
            claude_json: Claude API request as JSON string

        Returns:
            OpenAI API request as JSON string
        """
        claude_request = json.loads(claude_json)
        openai_request = self.convert_request(claude_request)
        return json.dumps(openai_request, indent=2)

    def process_folder(self, input_folder: str, output_folder: str = None):
        """
        Process all JSON files in the input folder and write converted files.

        Args:
            input_folder: Path to the folder containing Claude request JSON files
            output_folder: Path to the output folder (defaults to input_openai)
        """
        if output_folder is None:
            output_folder = input_folder + "_to_openai"

        # Find all JSON files recursively
        json_files = glob.glob(os.path.join(input_folder, "**/*.json"), recursive=True)

        print(f"Found {len(json_files)} JSON files to process")

        successful = 0
        failed = 0

        for json_file in json_files:
            try:
                # Read the Claude request
                with open(json_file, 'r', encoding='utf-8') as f:
                    claude_request = json.load(f)

                # Convert to OpenAI format
                openai_request = self.convert_request(claude_request)

                # Create output file path
                relative_path = os.path.relpath(json_file, input_folder)
                output_file = os.path.join(output_folder, relative_path)

                # Add _openai suffix to filename
                base_name = os.path.basename(output_file)
                dir_name = os.path.dirname(output_file)
                name_without_ext = os.path.splitext(base_name)[0]
                output_file = os.path.join(dir_name, f"{name_without_ext}_openai.json")

                # Create output directory if needed
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)

                # Write the converted request
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(openai_request, f, indent=2, ensure_ascii=False)

                print(f"✓ Converted: {relative_path} -> {os.path.relpath(output_file, output_folder)}")
                successful += 1

            except Exception as e:
                print(f"✗ Failed to convert {json_file}: {str(e)}")
                failed += 1

        print(f"\nConversion complete: {successful} successful, {failed} failed")
        print(f"Output folder: {output_folder}")


def main():
    """Process all JSON files in the specified input folder."""
    parser = argparse.ArgumentParser(description='Convert Claude API requests to OpenAI format')
    parser.add_argument('--input_folder', type=str, default='claude_requests',
                        help='Input folder containing Claude request JSON files (default: claude_requests)')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder for converted files (default: <input_folder>_to_openai)')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        print("Please make sure you're running the script from the correct directory.")
        return

    # Create converter instance
    converter = ClaudeToOpenAIConverter()

    # Process all files in the folder
    converter.process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
