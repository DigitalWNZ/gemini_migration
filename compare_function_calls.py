import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
import re
import argparse

def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file and return its contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_claude_function_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract function calls from Claude format."""
    function_calls = []
    if 'tool_calls' in data:
        for tool_call in data['tool_calls']:
            if tool_call.get('type') == 'function':
                function_calls.append({
                    'name': tool_call['function']['name'],
                    'arguments': tool_call['function']['arguments']
                })
    return function_calls

def extract_gemini_function_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract function calls from Gemini format or return finish_reason if no function calls."""
    function_calls = []
    if 'response' in data and 'content' in data['response']:
        content = data['response']['content']
        if 'parts' in content:
            for part in content['parts']:
                if 'functionCall' in part:
                    function_call = part['functionCall']
                    function_calls.append({
                        'name': function_call.get('name', ''),
                        'arguments': function_call.get('args', {})
                    })
    
    # If no function calls found, check for finish_reason
    if not function_calls and 'finish_reason' in data:
        function_calls.append({
            'name': f"[{data['finish_reason']}]",
            'arguments': {}
        })
    
    return function_calls

def extract_gemini_token_count(data: Dict[str, Any]) -> Optional[int]:
    """Extract candidates token count from Gemini response."""
    if 'usage' in data and 'candidatesTokenCount' in data['usage']:
        return data['usage']['candidatesTokenCount']
    return None

def format_arguments(args: Dict[str, Any]) -> str:
    """Format arguments as a readable string - show all parameters sorted by name."""
    if not args:
        return "{}"
    
    formatted_parts = []
    # Sort parameters by key name
    for key in sorted(args.keys()):
        value = args[key]
        if isinstance(value, str):
            # Only truncate the VALUE if it's very long, not the parameter name
            if len(value) > 100:
                value = value[:97] + "..."
        elif isinstance(value, list):
            # Show list details
            value = f"[{len(value)} items]"
        elif isinstance(value, dict):
            # Show dict details
            value = f"{{dict with {len(value)} keys}}"
        elif value is None:
            value = "None"
        
        # Always show the full parameter name
        formatted_parts.append(f"{key}: {value}")
    
    # Join all parameters with newlines for CSV
    return "\n".join(formatted_parts)

def compare_files(output_dir: str, output_gemini_dir: str) -> pd.DataFrame:
    """Compare function calls between Claude and Gemini outputs."""
    results = []
    
    # Get all subdirectories in output
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    for subdir in sorted(subdirs):
        claude_dir = os.path.join(output_dir, subdir)
        gemini_dir = os.path.join(output_gemini_dir, subdir)
        
        if not os.path.exists(gemini_dir):
            print(f"Warning: Gemini directory not found for {subdir}")
            continue
        
        # Get all step files in the directory
        step_files = [f for f in os.listdir(claude_dir) if f.startswith('step_') and f.endswith('.json')]
        
        for step_file in sorted(step_files):
            step_num = step_file.replace('step_', '').replace('.json', '')
            
            # Find all matching gemini files with iteration pattern
            gemini_files = []
            
            for f in os.listdir(gemini_dir):
                # Match files like step_2_gemini_1.json
                match = re.match(rf"step_{step_num}_gemini_(\d+)\.json", f)
                if match:
                    iteration = int(match.group(1))
                    gemini_files.append((f, iteration))
            
            # If no files with iteration pattern, try old format
            if not gemini_files:
                old_gemini_file = f"step_{step_num}_gemini.json"
                if os.path.exists(os.path.join(gemini_dir, old_gemini_file)):
                    gemini_files.append((old_gemini_file, 0))
            
            # Sort by iteration number
            gemini_files.sort(key=lambda x: x[1])
            
            claude_path = os.path.join(claude_dir, step_file)
            
            # Load Claude data once
            claude_data = load_json_file(claude_path)
            if not claude_data:
                continue
            
            # Extract Claude function calls once
            claude_calls = extract_claude_function_calls(claude_data)
            
            # Process each gemini file
            for gemini_file, iteration in gemini_files:
                gemini_path = os.path.join(gemini_dir, gemini_file)
                
                # Load Gemini data
                gemini_data = load_json_file(gemini_path)
                if not gemini_data:
                    continue
                
                # Extract Gemini function calls
                gemini_calls = extract_gemini_function_calls(gemini_data)
                
                # Extract token count
                token_count = extract_gemini_token_count(gemini_data)
                
                # Compare function calls
                max_calls = max(len(claude_calls), len(gemini_calls))
                
                for i in range(max_calls):
                    row = {
                        'File': f"output/{subdir}/step_{step_num}",
                        'Iteration': iteration if iteration > 0 else '',
                        'Claude Function': '',
                        'Claude Parameters': '',
                        'Gemini Function': '',
                        'Gemini Parameters': '',
                        'Match': '',
                        'Candidates Token Count': token_count if token_count is not None else ''
                    }
                    
                    if i < len(claude_calls):
                        row['Claude Function'] = claude_calls[i]['name']
                        row['Claude Parameters'] = format_arguments(claude_calls[i]['arguments'])
                    
                    if i < len(gemini_calls):
                        row['Gemini Function'] = gemini_calls[i]['name']
                        row['Gemini Parameters'] = format_arguments(gemini_calls[i]['arguments'])
                    
                    # Check if functions match
                    if row['Claude Function'] and row['Gemini Function']:
                        if row['Claude Function'] == row['Gemini Function']:
                            row['Match'] = 'Yes'
                        else:
                            row['Match'] = 'No'
                    else:
                        row['Match'] = 'Missing'
                    
                    results.append(row)
    
    return pd.DataFrame(results)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare function calls between Claude and Gemini outputs')
    parser.add_argument('--output-dir', 
                        default="agentic_data_demo/output",
                        help='Path to Claude output directory')
    parser.add_argument('--output-gemini-dir', 
                        default="agentic_data_demo/output_gemini",
                        help='Path to Gemini output directory')
    parser.add_argument('--output-csv', 
                        default="function_call_comparison.csv",
                        help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Get directories from arguments
    output_dir = args.output_dir
    output_gemini_dir = args.output_gemini_dir
    output_file = args.output_csv
    
    # Compare files and create DataFrame
    df = compare_files(output_dir, output_gemini_dir)
    
    # Save to CSV file
    df.to_csv(output_file, index=False)
    print(f"Comparison saved to: {output_file}")
    
    # Also save as a formatted text table
    base_name = os.path.splitext(output_file)[0]
    output_txt = f"{base_name}.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(df.to_string(index=False))
    print(f"Text table saved to: {output_txt}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total comparisons: {len(df)}")
    print(f"Matching functions: {len(df[df['Match'] == 'Yes'])}")
    print(f"Non-matching functions: {len(df[df['Match'] == 'No'])}")
    print(f"Missing functions: {len(df[df['Match'] == 'Missing'])}")

if __name__ == "__main__":
    main()