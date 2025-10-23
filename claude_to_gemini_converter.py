#!/usr/bin/env python3
"""
Claude to Gemini API Request Converter

This script converts API requests from Claude's format to Gemini's format.

1. Message Conversion

  - Roles mapping:
    - user → user (same)
    - assistant → model
    - system → extracted to systemInstruction (separate field)
    - tool → model (tool responses are treated as model responses)
  - Content structure:
    - Claude: {"role": "...", "content": "..."}
    - Gemini: {"role": "...", "parts": [{"text": "..."}]}

  2. System Messages

  - Claude format: Message with role: "system" in messages array
  - Gemini format:
  {
    "systemInstruction": {
      "parts": [{"text": "system message content"}]
    }
  }
    - Extracted from messages array and placed at top level

  3. Tool Calls (from assistant)

  - Claude format:
  {
    "role": "assistant",
    "content": [
      {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    ]
  }
  - Gemini format:
  {
    "role": "model",
    "parts": [{
      "functionCall": {
        "name": "...",
        "args": {...}
      }
    }]
  }
    - input → args
    - tool_use → functionCall

  4. Tool Results/Responses

  - Claude format:
  {
    "role": "user",
    "content": [
      {"type": "tool_result", "tool_use_id": "...", "content": "..."}
    ]
  }
  - Gemini format:
  {
    "role": "model",
    "parts": [{
      "functionResponse": {
        "name": "function_name",
        "response": {
          "result": "..."
        }
      }
    }]
  }
    - Function name is extracted from content using regex: Observation of Tool \{function_name}`
    - Content wrapped in response.result

  5. Tools/Functions Definition

  - Claude format:
  {
    "name": "...",
    "description": "...",
    "input_schema": {...}
  }
  - Gemini format:
  {
    "tools": [{
      "functionDeclarations": [{
        "name": "...",
        "description": "...",
        "parameters": {...}
      }]
    }]
  }
    - Tools are wrapped in functionDeclarations array
    - input_schema → parameters

  6. Request Structure

  - Claude: {"messages": [...], "tools": [...], "model": "...", ...}
  - Gemini: {"contents": [...], "systemInstruction": {...}, "tools": [...]}
    - messages → contents
    - System messages extracted to systemInstruction
    - Model and other parameters not included
"""

import json
import os
import glob
import copy
import re
from typing import Dict, List, Any, Optional


class ClaudeToGeminiConverter:
    """Converts Claude API requests to Gemini API format."""
    
    # Model mapping from Claude to Gemini
    MODEL_MAPPING = {
        "claude-3-opus-20240229": "gemini-1.5-pro",
        "claude-3-sonnet-20240229": "gemini-1.5-flash",
        "claude-3-haiku-20240307": "gemini-1.5-flash",
        "claude-3-5-sonnet-20241022": "gemini-1.5-pro",
        "claude-3-5-haiku-20241022": "gemini-1.5-flash",
        # Add more mappings as needed
    }
    
    def __init__(self):
        """Initialize the converter."""
        pass
    
    def convert_messages(self, claude_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Claude message format to Gemini format.
        
        Claude format: {"role": "user/assistant/tool", "content": "text"}
        Gemini format: {"role": "user/model", "parts": [{"text": "text"}]}
        """
        gemini_messages = []
        
        for message in claude_messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Map Claude roles to Gemini roles
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            elif role == "tool":
                # Tool responses should be treated as model responses
                gemini_role = "model"
                # Format tool response
                tool_name = message.get("name", "tool")
                tool_content = f"Tool Response ({tool_name}):\n{content}"
                parts = [{"text": tool_content}]
                gemini_messages.append({
                    "role": gemini_role,
                    "parts": parts
                })
                continue
            elif role == "system":
                # System messages are handled separately in convert_request
                continue
            else:
                gemini_role = "user"  # Default fallback
            
            # Convert content to parts format
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                # Handle multi-modal content
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append({"text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image":
                            # Convert image format
                            parts.append({
                                "inline_data": {
                                    "mime_type": item.get("source", {}).get("media_type", "image/jpeg"),
                                    "data": item.get("source", {}).get("data", "")
                                }
                            })
                        elif item.get("type") == "tool_use":
                            # Handle function calls
                            # Extract the "name" and "input" object
                            function_name = item.get('name', 'unknown')
                            function_args = item.get('input', {})
                            # Change "input" to "args" and embrace both in "functionCall" object
                            parts.append({
                                "functionCall": {
                                    "name": function_name,
                                    "args": function_args
                                }
                            })
                        elif item.get("type") == "tool_result":
                            # Handle tool results
                            # Extract the "content" object
                            content = item.get('content', '')
                            # Extract function name from the content value, the pattern is "Observation of Tool `{function name}`"
                            match = re.search(r'Observation of Tool `([^`]+)`', content)
                            function_name = match.group(1) if match else 'unknown'
                            # Change "content" to "response" and embrace both name and response in "functionResponse" object
                            parts.append({
                                "functionResponse": {
                                    "name": function_name,
                                    "response": {
                                        "result": content
                                    }
                                }
                            })
            else:
                parts = [{"text": str(content)}]
            
            gemini_messages.append({
                "role": gemini_role,
                "parts": parts
            })
        
        return gemini_messages
    
    def convert_tools(self, claude_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Claude tools (functions) to Gemini function declarations.
        
        Claude format: tools with name, description, input_schema
        Gemini format: functionDeclarations with name, description, parameters
        """
        if not claude_tools:
            return []
        
        gemini_functions = []
        for tool in claude_tools:
            # Convert input_schema to parameters format
            input_schema = tool.get("input_schema", {})
            
            gemini_function = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": input_schema  # Gemini uses the same JSON Schema format
            }
            
            gemini_functions.append(gemini_function)
        
        return gemini_functions
    
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
    
    def convert_request(self, claude_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a complete Claude API request to Gemini format.
        
        Args:
            claude_request: Claude API request dictionary
            
        Returns:
            Gemini API request dictionary
        """
        # Extract only the required fields
        messages = claude_request.get("messages", [])
        tools = claude_request.get("tools", [])
        
        # Extract system messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # Convert non-system messages
        gemini_contents = self.convert_messages(non_system_messages)
        
        # Build Gemini request with only message and tools fields
        gemini_request = {
            "contents": gemini_contents
        }
        
        # Add system instruction if present
        if system_messages:
            system_text = "\n".join([msg.get("content", "") for msg in system_messages])
            gemini_request["systemInstruction"] = {
                "parts": [{"text": system_text}]
            }
        
        # Convert and add tools if present
        if tools:
            gemini_functions = self.convert_tools(tools)
            if gemini_functions:
                gemini_request["tools"] = [{
                    "functionDeclarations": gemini_functions
                }]
        
        # Apply fixes to the converted request
        gemini_request = self.fix_request_issues(gemini_request)
        
        return gemini_request
    
    def convert_from_json(self, claude_json: str) -> str:
        """
        Convert Claude request from JSON string to Gemini JSON string.
        
        Args:
            claude_json: Claude API request as JSON string
            
        Returns:
            Gemini API request as JSON string
        """
        claude_request = json.loads(claude_json)
        gemini_request = self.convert_request(claude_request)
        return json.dumps(gemini_request, indent=2)


    def process_folder(self, input_folder: str, output_folder: str = None):
        """
        Process all JSON files in the input folder and write converted files.
        
        Args:
            input_folder: Path to the folder containing Claude request JSON files
            output_folder: Path to the output folder (defaults to input_gemini)
        """
        if output_folder is None:
            # Extract the parent directory of input_folder
            parent_dir = os.path.dirname(input_folder)
            output_folder = os.path.join(parent_dir, "input_gemini")
        
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
                
                # Convert to Gemini format
                gemini_request = self.convert_request(claude_request)
                
                # Create output file path
                relative_path = os.path.relpath(json_file, input_folder)
                output_file = os.path.join(output_folder, relative_path)
                
                # Add _gemini suffix to filename
                base_name = os.path.basename(output_file)
                dir_name = os.path.dirname(output_file)
                name_without_ext = os.path.splitext(base_name)[0]
                output_file = os.path.join(dir_name, f"{name_without_ext}_gemini.json")
                
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
    """Process all JSON files in the agentic_data_demo/input folder."""
    input_folder = "agentic_data_demo/input"
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        print("Please make sure you're running the script from the correct directory.")
        return
    
    # Create converter instance
    converter = ClaudeToGeminiConverter()
    
    # Process all files in the folder
    converter.process_folder(input_folder)
    
    # Example of converting a single file
    print("\n" + "="*50 + "\n")
    print("Example of single file conversion:")
    
    # Get the output folder path
    output_folder = "agentic_data_demo/input_gemini_v1"
    sample_files = glob.glob(os.path.join(output_folder, "**/*_gemini.json"), recursive=True)
    if sample_files:
        sample_file = sample_files[0]
        print(f"\nConverted sample file: {sample_file}")
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            gemini_request = json.load(f)
        
        print("\nConverted Gemini Request:")
        print(json.dumps(gemini_request, indent=2))


if __name__ == "__main__":
    main()