#!/usr/bin/env python3
"""
Gemini API Caller using REST API

This script reads Gemini-formatted requests from input_gemini folder,
calls the Gemini API using REST endpoints, and stores results in output_gemini folder.
"""

import json
import os
import time
import copy
from typing import Dict, Any
import requests
import argparse


class GeminiAPICaller:
    """Calls Gemini API with converted requests."""
    
    def __init__(self, credentials=None, fc2=True, function_call_mode="auto"):
        """
        Initialize the Gemini API caller.
        
        Args:
            credentials: Google Cloud credentials object (optional)
            fc2: Use cloud-llm-preview4 if True, cloud-llm-preview1 if False (default: True)
            function_call_mode: Function calling mode - "auto", "any", "validated" (default: "auto")
        """
        # if fc2:
        #     if function_call_mode == "validated":
        #         self.base_url = "https://aiplatform.googleapis.com/v1beta1/projects/cloud-llm-preview4/locations/global/publishers/google/models/gemini-2.5-pro:generateContent"
        #     else:
        #         self.base_url = "https://aiplatform.googleapis.com/v1/projects/cloud-llm-preview4/locations/global/publishers/google/models/gemini-2.5-pro:generateContent"
        # else:
        #     self.base_url = "https://aiplatform.googleapis.com/v1/projects/cloud-llm-preview1/locations/global/publishers/google/models/gemini-2.5-pro:generateContent"
        if fc2:
            if function_call_mode == "validated":
                self.base_url = "https://aiplatform.googleapis.com/v1beta1/projects/agent-sp-474006/locations/global/publishers/google/models/gemini-2.5-pro:generateContent"
            else:
                self.base_url = "https://aiplatform.googleapis.com/v1/projects/agent-sp-474006/locations/global/publishers/google/models/gemini-2.5-pro:generateContent"
        else:
            self.base_url = "https://aiplatform.googleapis.com/v1/projects/agent-sp-474006/locations/global/publishers/google/models/gemini-2.5-pro:generateContent"
        self.credentials = credentials
        self.api_key = None  # Will be set when needed
        self.fc2 = fc2  # Store fc2 setting for labels
        
    def fix_request_issues(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix known issues in the request data before sending to Gemini API."""
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
    
    def get_access_token(self):
        """Get access token from credentials."""
        if self.credentials:
            # Refresh the credentials if needed
            import google.auth.transport.requests
            request = google.auth.transport.requests.Request()
            self.credentials.refresh(request)
            return self.credentials.token
        else:
            # Fallback to hardcoded token if no credentials
            return self.api_key
    
    def call_gemini(self, request: Dict[str, Any], function_call_mode: str = "auto", thinking_budget: int = 0, 
                    session_id: str = None, iteration: int = 1, step_id: str = None) -> Dict[str, Any]:
        """
        Call Gemini API with the given request using REST API.
        
        Args:
            request: Gemini-formatted request dictionary
            function_call_mode: Function calling mode - "auto", "any", "validated" (default: "auto")
            thinking_budget: Thinking budget value. If 0 or not set, generationConfig is not added (default: 0)
            session_id: The session ID (subfolder name) for labeling
            iteration: The iteration number for labeling
            step_id: The step ID (filename without extension) for labeling
            
        Returns:
            Response dictionary
        """
        try:
            # Fix known issues in the request before processing
            # fixed_request = self.fix_request_issues(request)
            
            # Add toolConfig based on function_call_mode
            if 'tools' in request and function_call_mode != "auto":
                request['toolConfig'] = {
                    "functionCallingConfig": {
                        "mode": function_call_mode.upper()
                    }
                }
            
            # Add generationConfig with thinking budget if specified
            if thinking_budget > 0:
                request['generationConfig'] = {
                    "thinkingConfig": {
                        "thinkingBudget": thinking_budget
                    }
                }
            
            # Add labels with timestamp, session_id, step_id, iteration, and configuration settings
            if session_id:
                request['labels'] = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "session_id": session_id,
                    "step_id": step_id if step_id else "",
                    "iteration": str(iteration),
                    "fc2": str(self.fc2).lower(),
                    "function_call_mode": function_call_mode,
                    "thinking_budget": str(thinking_budget)
                }
            
            # Get access token
            access_token = self.get_access_token()
            
            # Add API key to the URL
            url = self.base_url
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            # Make the API request
            response = requests.post(
                url,
                headers=headers,
                json=request,
                timeout=120  # 2 minute timeout
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the generated text from the response
            generated_text = ""
            finish_reason = "STOP"
            
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                
                # Extract text from content parts
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            generated_text += part["text"]
                
                # Get finish reason
                if "finishReason" in candidate:
                    finish_reason = candidate["finishReason"]
            
            # Check for prompt feedback issues
            if "promptFeedback" in response_data:
                feedback = response_data["promptFeedback"]
                if "blockReason" in feedback:
                    return {
                        "error": f"Prompt was blocked: {feedback['blockReason']}",
                        "error_type": "PromptBlocked",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                        "raw_response": response_data
                    }
            
            # Format successful response
            result = {
                "response": response_data["candidates"][0],
                "model": "gemini-2.5-pro",
                "finish_reason": finish_reason,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                # "raw_response":response_data
            }
            
            # Add usage metadata if available
            if "usageMetadata" in response_data:
                result["usage"] = response_data["usageMetadata"]
            
            return result
            
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors
            error_message = f"HTTP {e.response.status_code}: {e.response.text}"
            return {
                "error": error_message,
                "error_type": "HTTPError",
                "status_code": e.response.status_code,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out after 120 seconds",
                "error_type": "Timeout",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
        except Exception as e:
            # Return error response for any other exception
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
    
    def process_folder(self, input_folder: str, output_folder: str, iteration: int = 1, function_call_mode: str = "auto", thinking_budget: int = 0):
        """
        Process all JSON files in the input folder and save results.
        Processes subfolder by subfolder, with files in alphabetical order.
        
        Args:
            input_folder: Path to folder containing Gemini request JSON files
            output_folder: Path to output folder for results
            iteration: Current iteration number
            function_call_mode: Function calling mode - "auto", "any", "validated" (default: "auto")
            thinking_budget: Thinking budget value. If 0 or not set, generationConfig is not added (default: 0)
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all subfolders
        subfolders = []
        for root, _, files in os.walk(input_folder):
            # Skip the root folder if it has no JSON files
            json_files_in_folder = [f for f in files if f.endswith('.json')]
            if json_files_in_folder:
                subfolders.append(root)
        
        # Sort subfolders alphabetically
        subfolders.sort()
        
        print(f"Found {len(subfolders)} folders to process")
        
        total_successful = 0
        total_failed = 0
        total_files = 0
        
        # Process each subfolder
        for folder_idx, subfolder in enumerate(subfolders, 1):
            # Get relative subfolder path for display
            relative_subfolder = os.path.relpath(subfolder, input_folder)
            if relative_subfolder == ".":
                relative_subfolder = "root"
            
            print(f"\n{'='*60}")
            print(f"Processing subfolder {folder_idx}/{len(subfolders)}: {relative_subfolder}")
            print(f"{'='*60}")
            
            # Find all JSON files in this subfolder (not recursive)
            json_files = []
            for file in os.listdir(subfolder):
                if file.endswith('.json'):
                    json_files.append(os.path.join(subfolder, file))
            
            # Sort files alphabetically
            json_files.sort()
            
            print(f"Found {len(json_files)} JSON files in this subfolder")
            
            successful = 0
            failed = 0
            
            # Process each file in the subfolder
            for i, json_file in enumerate(json_files, 1):
                try:
                    print(f"\n  Processing file {i}/{len(json_files)}: {os.path.basename(json_file)}")
                    
                    # Read the Gemini request
                    with open(json_file, 'r', encoding='utf-8') as f:
                        gemini_request = json.load(f)
                    
                    # Extract session_id from the subfolder path
                    # The subfolder name is the last part of the path (e.g., "0f6e4002-149c-4105-8299-2c4b364908a6_3602")
                    session_id = os.path.basename(subfolder)
                    
                    # Extract step_id from the filename (e.g., "step_0_gemini.json" -> "step_0_gemini")
                    step_id = os.path.splitext(os.path.basename(json_file))[0]
                    
                    # Call Gemini API
                    print("    Calling Gemini API...")
                    result = self.call_gemini(gemini_request.copy(), function_call_mode, thinking_budget, session_id, iteration, step_id)  # Use copy to preserve original
                    
                    # Create output file path with iteration number
                    relative_path = os.path.relpath(json_file, input_folder)
                    # Add iteration number to filename
                    base_name, ext = os.path.splitext(relative_path)
                    output_filename = f"{base_name}_{iteration}{ext}"
                    output_file = os.path.join(output_folder, output_filename)
                    
                    # Create output directory if needed
                    output_dir = os.path.dirname(output_file)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Write the result
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    if "error" in result:
                        print(f"    ✗ API call failed: {result['error']}")
                        failed += 1
                    else:
                        print(f"    ✓ Success: {relative_path}")
                        successful += 1
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"    ✗ Failed to process {json_file}: {str(e)}")
                    failed += 1
            
            # Update totals
            total_successful += successful
            total_failed += failed
            total_files += len(json_files)
            
            print(f"\nSubfolder summary: {successful} successful, {failed} failed")
        
        print(f"\n{'='*60}")
        print(f"Overall processing complete:")
        print(f"  Total files: {total_files}")
        print(f"  Successful: {total_successful}")
        print(f"  Failed: {total_failed}")
        print(f"{'='*60}")


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """Main function to process Gemini requests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Gemini API requests')
    parser.add_argument('--input-folder', default="agentic_data_demo/input_gemini",
                        help='Input folder containing Gemini request JSON files')
    parser.add_argument('--output-folder', default="agentic_data_demo/output_gemini",
                        help='Output folder for results')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations to process all requests')
    parser.add_argument('--function-call-mode', type=str, default="auto",
                        choices=["auto", "any", "validated"],
                        help='Function calling mode (default: auto)')
    parser.add_argument('--fc2', type=str2bool, default=False,
                        help='Use cloud-llm-preview4 project (default: False, uses cloud-llm-preview1)')
    parser.add_argument('--thinking-budget', type=int, default=0,
                        help='Thinking budget value. If 0, generationConfig is not added (default: 0)')
    
    args = parser.parse_args()
    
    import google.auth
    credentials, _ = google.auth.default()

    input_folder = args.input_folder
    output_folder = args.output_folder
    iterations = args.iterations
    function_call_mode = args.function_call_mode
    fc2 = args.fc2
    thinking_budget = args.thinking_budget
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        print("Please make sure the input_gemini folder exists with converted requests.")
        return
    
    # Create API caller instance with credentials
    caller = GeminiAPICaller(credentials=credentials, fc2=fc2, function_call_mode=function_call_mode)
    
    # Process all files for each iteration
    print(f"Processing files from: {input_folder}")
    print(f"Output will be saved to: {output_folder}")
    print(f"Number of iterations: {iterations}")
    print(f"Function call mode: {function_call_mode}")
    print(f"FC2 mode: {fc2}")
    print(f"Thinking budget: {thinking_budget}")
    print("Using Gemini 2.5 Pro via AI Platform endpoint")
    
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*80}")
        print(f"STARTING ITERATION {iteration} of {iterations}")
        print(f"{'='*80}")
        caller.process_folder(input_folder, output_folder, iteration, function_call_mode, thinking_budget)


if __name__ == "__main__":
    main()