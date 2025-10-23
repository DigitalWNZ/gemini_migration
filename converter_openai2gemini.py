#!/usr/bin/env python3
"""
OpenAI to Gemini API Request Converter

This script converts API requests from OpenAI's format to Gemini's format.

1. Converts messages:
- OpenAI system role → Gemini systemInstruction
- OpenAI user role → Gemini user role
- OpenAI assistant role → Gemini model role
- OpenAI tool role → Gemini user role with functionResponse
2. Converts tool calls:
- OpenAI tool_calls with function → Gemini functionCall with name and args
3. Converts tool responses:
- OpenAI tool message with name and content → Gemini functionResponse with name and response.result
4. Converts tools:
- OpenAI tools with type: "function" and nested function → Gemini tools with functionDeclarations
"""

import argparse
import json
import os
import glob
import copy
import re
from typing import Dict, List, Any, Optional


class OpenAIToGeminiConverter:
    """Converts OpenAI API requests to Gemini API format."""

    def __init__(self):
        """Initialize the converter."""
        pass

    def convert_messages(self, openai_messages: List[Dict[str, Any]]) -> tuple:
        """
        Convert OpenAI message format to Gemini format.

        OpenAI format: {"role": "user/assistant/system/tool", "content": "text"}
        Gemini format: {"role": "user/model", "parts": [{"text": "text"}]}

        Returns:
            Tuple of (contents list, system_instruction dict or None)
        """
        gemini_contents = []
        system_instruction = None

        # Build a mapping of tool_call_id to tool_name from tool_calls messages
        tool_call_id_to_name = {}
        for message in openai_messages:
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    tool_call_id = tool_call.get("id", "")
                    function_data = tool_call.get("function", {})
                    tool_name = function_data.get("name", "")
                    if tool_call_id and tool_name:
                        tool_call_id_to_name[tool_call_id] = tool_name

        for message in openai_messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Handle system messages separately
            if role == "system":
                if isinstance(content, str):
                    system_instruction = {
                        "parts": [{"text": content}]
                    }
                continue

            # Map OpenAI roles to Gemini roles
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            elif role == "tool":
                # Tool responses map to user role with functionResponse
                gemini_role = "user"
            else:
                gemini_role = "user"  # Default fallback

            # Handle tool calls (from assistant)
            if "tool_calls" in message and message["tool_calls"]:
                parts = []
                for tool_call in message["tool_calls"]:
                    function_data = tool_call.get("function", {})
                    args = function_data.get("arguments", "{}")
                    # Parse arguments if it's a string
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}

                    parts.append({
                        "functionCall": {
                            "name": function_data.get("name", ""),
                            "args": args
                        }
                    })

                gemini_contents.append({
                    "role": gemini_role,
                    "parts": parts
                })
                continue

            # Handle tool responses
            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                # Try to get tool_name from message first, then from tool_call_id mapping
                tool_name = message.get("name", "")
                if not tool_name and tool_call_id:
                    tool_name = tool_call_id_to_name.get(tool_call_id, "")
                tool_content = content if isinstance(content, str) else str(content)

                parts = [{
                    "functionResponse": {
                        "name": tool_name,
                        "response": {
                            "result": tool_content
                        }
                    }
                }]

                gemini_contents.append({
                    "role": gemini_role,
                    "parts": parts
                })
                continue

            # Convert regular content
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                # Handle multi-modal content
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Convert image format
                            image_url_data = item.get("image_url", {})
                            url = image_url_data.get("url", "")

                            # Handle base64 encoded images
                            if url.startswith("data:"):
                                # Extract mime type and base64 data
                                match = re.match(r'data:([^;]+);base64,(.+)', url)
                                if match:
                                    mime_type = match.group(1)
                                    base64_data = match.group(2)
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": base64_data
                                        }
                                    })
                            else:
                                # URL reference
                                parts.append({"text": f"[Image: {url}]"})
                    elif isinstance(item, str):
                        parts.append({"text": item})
            else:
                parts = [{"text": str(content)}]

            if parts:
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": parts
                })

        return gemini_contents, system_instruction

    def convert_enum_values(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert integer enum values to strings in parameter schemas.

        OpenAI format: "enum": [1, 2], "type": "integer"
        Gemini format: "enum": ["1", "2"], "type": "integer"
        """
        if not isinstance(schema, dict):
            return schema

        # Create a copy to avoid modifying the original
        schema_copy = copy.deepcopy(schema)

        # Convert enum values if present
        if "enum" in schema_copy and isinstance(schema_copy["enum"], list):
            schema_copy["enum"] = [str(val) for val in schema_copy["enum"]]

        # Recursively process nested schemas
        if "properties" in schema_copy and isinstance(schema_copy["properties"], dict):
            schema_copy["properties"] = {
                key: self.convert_enum_values(val)
                for key, val in schema_copy["properties"].items()
            }

        if "items" in schema_copy:
            schema_copy["items"] = self.convert_enum_values(schema_copy["items"])

        if "additionalProperties" in schema_copy and isinstance(schema_copy["additionalProperties"], dict):
            schema_copy["additionalProperties"] = self.convert_enum_values(schema_copy["additionalProperties"])

        return schema_copy

    def convert_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI tools to Gemini function declarations.

        OpenAI format: [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
        Gemini format: [{"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}]
        """
        if not openai_tools:
            return []

        function_declarations = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                function_data = tool.get("function", {})
                parameters = self.convert_enum_values(function_data.get("parameters", {}))
                function_declarations.append({
                    "name": function_data.get("name", ""),
                    "description": function_data.get("description", ""),
                    "parameters": parameters
                })

        if function_declarations:
            return [{
                "functionDeclarations": function_declarations
            }]

        return []

    def fix_request_issues(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix known issues in the request data during conversion."""
        # Deep copy to avoid modifying the original
        fixed_data = copy.deepcopy(request_data)

        # Fix empty parts issue
        if 'contents' in fixed_data and isinstance(fixed_data['contents'], list):
            # Filter out messages with empty parts
            fixed_data['contents'] = [
                content for content in fixed_data['contents']
                if 'parts' in content and content['parts'] and len(content['parts']) > 0
            ]

        # Check if tools exist in the request
        if 'tools' in fixed_data and isinstance(fixed_data['tools'], list):
            for tool in fixed_data['tools']:
                if 'functionDeclarations' in tool and isinstance(tool['functionDeclarations'], list):
                    for func in tool['functionDeclarations']:
                        # Fix parameter mismatches
                        if func.get('name') == 'segment_anything':
                            if 'required' in func['parameters'] and 'object' in func['parameters']['required']:
                                # Replace 'object' with 'object_english_name' in required array
                                func['parameters']['required'] = [
                                    'object_english_name' if param == 'object' else param
                                    for param in func['parameters']['required']
                                ]

                        elif func.get('name') == 'Pira_image2image':
                            if 'required' in func['parameters'] and 'cfg' in func['parameters']['required']:
                                # Remove 'cfg' from required array as it doesn't exist in properties
                                func['parameters']['required'] = [
                                    param for param in func['parameters']['required']
                                    if param != 'cfg'
                                ]

                        elif func.get('name') == 'gemini_edit':
                            # Fix required parameter name
                            if 'required' in func['parameters'] and 'image' in func['parameters']['required']:
                                # Replace 'image' with 'images' in required array
                                func['parameters']['required'] = [
                                    'images' if param == 'image' else param
                                    for param in func['parameters']['required']
                                ]

                            # Fix the type of images field from ["array", "null"] to "array"
                            if 'properties' in func['parameters'] and 'images' in func['parameters']['properties']:
                                if isinstance(func['parameters']['properties']['images'].get('type'), list):
                                    func['parameters']['properties']['images']['type'] = 'array'

                        elif func.get('name') == 'outpaint':
                            if 'required' in func['parameters'] and 'prompt' in func['parameters']['required']:
                                # Replace 'prompt' with 'english_prompt' in required array
                                func['parameters']['required'] = [
                                    'english_prompt' if param == 'prompt' else param
                                    for param in func['parameters']['required']
                                ]

        return fixed_data

    def convert_request(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a complete OpenAI API request to Gemini format.

        Args:
            openai_request: OpenAI API request dictionary

        Returns:
            Gemini API request dictionary
        """
        # Extract fields
        messages = openai_request.get("messages", [])
        tools = openai_request.get("tools", [])

        # Convert messages
        gemini_contents, system_instruction = self.convert_messages(messages)

        # Build Gemini request
        gemini_request = {
            "contents": gemini_contents
        }

        # Add system instruction if present
        if system_instruction:
            gemini_request["systemInstruction"] = system_instruction

        # Convert and add tools if present
        if tools:
            gemini_tools = self.convert_tools(tools)
            if gemini_tools:
                gemini_request["tools"] = gemini_tools

        # Apply fixes to the converted request
        gemini_request = self.fix_request_issues(gemini_request)

        return gemini_request

    def convert_from_json(self, openai_json: str) -> str:
        """
        Convert OpenAI request from JSON string to Gemini JSON string.

        Args:
            openai_json: OpenAI API request as JSON string

        Returns:
            Gemini API request as JSON string
        """
        openai_request = json.loads(openai_json)
        gemini_request = self.convert_request(openai_request)
        return json.dumps(gemini_request, indent=2)

    def process_folder(self, input_folder: str, output_folder: str = None):
        """
        Process all JSON files in the input folder and write converted files.

        Args:
            input_folder: Path to the folder containing OpenAI request JSON files
            output_folder: Path to the output folder (defaults to input_gemini_from_openai)
        """
        if output_folder is None:
            output_folder = input_folder + "_to_gemini"

        # Find all JSON files recursively
        json_files = glob.glob(os.path.join(input_folder, "**/*.json"), recursive=True)

        print(f"Found {len(json_files)} JSON files to process")

        successful = 0
        failed = 0

        for json_file in json_files:
            try:
                # Read the OpenAI request
                with open(json_file, 'r', encoding='utf-8') as f:
                    openai_request = json.load(f)

                # Convert to Gemini format
                gemini_request = self.convert_request(openai_request)

                # Create output file path
                relative_path = os.path.relpath(json_file, input_folder)
                output_file = os.path.join(output_folder, relative_path)

                # Replace _openai suffix with _gemini_from_openai
                base_name = os.path.basename(output_file)
                dir_name = os.path.dirname(output_file)
                name_without_ext = os.path.splitext(base_name)[0]

                # Remove _openai suffix if present
                if name_without_ext.endswith("_openai"):
                    name_without_ext = name_without_ext[:-len("_openai")]

                output_file = os.path.join(dir_name, f"{name_without_ext}_gemini_from_openai.json")

                # Create output directory if needed
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)

                # Write the converted request
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(gemini_request, f, indent=2, ensure_ascii=False)

                print(f"✓ Converted: {relative_path} -> {os.path.relpath(output_file, output_folder)}")
                successful += 1

            except Exception as e:
                print(f"✗ Failed to convert {json_file}: {str(e)}")
                failed += 1

        print(f"\nConversion complete: {successful} successful, {failed} failed")
        print(f"Output folder: {output_folder}")


def main():
    """Process all JSON files in the specified input folder."""
    parser = argparse.ArgumentParser(description='Convert OpenAI API requests to Gemini format')
    parser.add_argument('--input_folder', type=str, default='openai_request',
                        help='Input folder containing OpenAI request JSON files (default: openai_request)')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder for converted files (default: <input_folder>_to_gemini)')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        print("Please make sure you're running the script from the correct directory.")
        return

    # Create converter instance
    converter = OpenAIToGeminiConverter()

    # Process all files in the folder
    converter.process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
