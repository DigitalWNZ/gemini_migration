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
echo "1. Running FC1 mode (cloud-llm-preview1, function-call-mode: any)..."
python gemini_api_caller.py --output-folder agentic_data_demo/output_gemini_FC1 \
                           --fc2 false \
                           --function-call-mode auto \
                           --iterations $ITERATIONS

echo ""
echo "2. Running FC2 mode (cloud-llm-preview4, function-call-mode: any)..."
python gemini_api_caller.py --output-folder agentic_data_demo/output_gemini_FC2 \
                           --fc2 true \
                           --function-call-mode auto \
                           --iterations $ITERATIONS

echo ""
echo "3. Running FC2 validated mode (cloud-llm-preview4, function-call-mode: validated)..."
python gemini_api_caller.py --output-folder agentic_data_demo/output_gemini_FC2_validate \
                           --fc2 true \
                           --function-call-mode validated \
                           --iterations $ITERATIONS

echo ""
echo "=========================================="
echo "API calls completed. Running comparisons..."
echo "=========================================="
echo ""

# Run comparison scripts
echo "4. Comparing FC1 results..."
python compare_function_calls.py --output-gemini-dir agentic_data_demo/output_gemini_FC1 \
                                --output-csv function_call_comparison_FC1.csv

echo ""
echo "5. Comparing FC2 results..."
python compare_function_calls.py --output-gemini-dir agentic_data_demo/output_gemini_FC2 \
                                --output-csv function_call_comparison_FC2.csv

echo ""
echo "6. Comparing FC2 validated results..."
python compare_function_calls.py --output-gemini-dir agentic_data_demo/output_gemini_FC2_validate \
                                --output-csv function_call_comparison_FC2_validate.csv

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - agentic_data_demo/output_gemini_FC1/"
echo "  - agentic_data_demo/output_gemini_FC2/"
echo "  - agentic_data_demo/output_gemini_FC2_validate/"
echo ""
echo "Comparison reports:"
echo "  - function_call_comparison_FC1.csv"
echo "  - function_call_comparison_FC2.csv"
echo "  - function_call_comparison_FC2_validate.csv"