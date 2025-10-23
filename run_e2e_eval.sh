#!/bin/bash

# Script to run Gemini API tests with different configurations
# Usage: ./run_gemini_tests.sh <iterations>

# Check if iteration parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide the number of iterations as a parameter"
    echo "Usage: $0 <iterations>"
    echo "Example: $0 2"
    exit 1
fi

# Get the number of iterations from command line argument
ITERATIONS=$1

# Validate that iterations is a positive integer
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -eq 0 ]; then
    echo "Error: Iterations must be a positive integer"
    exit 1
fi

echo "=========================================="
echo "Running Gemini API tests with $ITERATIONS iterations"
echo "=========================================="
echo ""

# Run Gemini API caller with different configurations
echo "1. Running FC1 mode (cloud-llm-preview1, function-call-mode: auto)..."
python gemini_api_caller.py --input-folder claude_requests_to_gemini \
                           --output-folder claude_requests_to_gemini_output_FC1 \
                           --iterations $ITERATIONS \
                           --function-call-mode auto \
                           --fc2 false \
                           --thinking-budget 0 \
                           --project cloud-llm-preview4 \
                           --model-name gemini-2.5-pro \
                           --openai-endpoint false \
                           --location global

echo ""
echo "2. Running FC2 mode (cloud-llm-preview4, function-call-mode: auto)..."
python gemini_api_caller.py --input-folder claude_requests_to_gemini \
                           --output-folder claude_requests_to_gemini_output_FC2 \
                           --iterations $ITERATIONS \
                           --function-call-mode auto \
                           --fc2 true \
                           --thinking-budget 0 \
                           --project cloud-llm-preview4 \
                           --model-name gemini-2.5-pro \
                           --openai-endpoint false \
                           --location global

echo ""
echo "3. Running FC2 validated mode (cloud-llm-preview4, function-call-mode: validated)..."
python gemini_api_caller.py --input-folder claude_requests_to_gemini \
                           --output-folder claude_requests_to_gemini_output_FC2_validate \
                           --iterations $ITERATIONS \
                           --function-call-mode validated \
                           --fc2 true \
                           --thinking-budget 0 \
                           --project cloud-llm-preview4 \
                           --model-name gemini-2.5-pro \
                           --openai-endpoint false \
                           --location global

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - claude_requests_to_gemini_output_FC1/"
echo "  - claude_requests_to_gemini_output_FC2/"
echo "  - claude_requests_to_gemini_output_FC2_validate/"