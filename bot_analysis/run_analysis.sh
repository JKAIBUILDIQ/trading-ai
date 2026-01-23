#!/bin/bash
#
# Crellastein Bot Analysis Runner
# Runs analysis on multiple LLMs and aggregates results
#

set -e

ANALYSIS_DIR="$HOME/trading_ai/bot_analysis"
cd "$ANALYSIS_DIR"

echo "============================================================"
echo "CRELLASTEIN BOT ANALYSIS - Multi-LLM Review"
echo "============================================================"
echo "Date: $(date)"
echo "Directory: $ANALYSIS_DIR"
echo ""

# Step 1: Build the full prompt
echo "📝 Building analysis prompt..."
cat llm_analysis_prompt.txt > full_prompt.txt
echo "" >> full_prompt.txt
echo "=== CURRENT BOT PARAMETERS ===" >> full_prompt.txt
echo "" >> full_prompt.txt

for bot_file in crellastein_*.json; do
    echo "--- $bot_file ---" >> full_prompt.txt
    cat "$bot_file" >> full_prompt.txt
    echo "" >> full_prompt.txt
done

echo "✅ Full prompt built ($(wc -l < full_prompt.txt) lines)"
echo ""

# Step 2: Check available Ollama models
echo "🔍 Checking available Ollama models..."
if command -v ollama &> /dev/null; then
    AVAILABLE_MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || echo "")
    if [ -n "$AVAILABLE_MODELS" ]; then
        echo "Available models:"
        echo "$AVAILABLE_MODELS" | while read model; do echo "   - $model"; done
    else
        echo "⚠️  No models found. Pull models with: ollama pull llama2:70b"
    fi
else
    echo "⚠️  Ollama not installed. Install from: https://ollama.ai"
fi
echo ""

# Step 3: Run analysis on available models
echo "🤖 Running LLM Analysis..."
echo ""

run_llm_analysis() {
    local model=$1
    local output_file=$2
    
    echo "   Running $model..."
    if ollama run "$model" < full_prompt.txt > "$output_file" 2>/dev/null; then
        echo "   ✅ $model analysis complete -> $output_file"
    else
        echo "   ❌ $model failed or not available"
    fi
}

# Try to run on available models
# Uncomment the models you have available

# Large models (best quality)
# run_llm_analysis "llama2:70b" "analysis_llama2_70b.json"
# run_llm_analysis "mixtral:8x7b" "analysis_mixtral.json"
# run_llm_analysis "codellama:34b" "analysis_codellama.json"
# run_llm_analysis "deepseek-coder:33b" "analysis_deepseek.json"

# Medium models (faster)
# run_llm_analysis "llama2:13b" "analysis_llama2_13b.json"
# run_llm_analysis "mistral:7b" "analysis_mistral.json"
# run_llm_analysis "codellama:7b" "analysis_codellama_7b.json"

# Small models (fastest)
# run_llm_analysis "llama2:7b" "analysis_llama2_7b.json"

echo ""
echo "============================================================"
echo "📊 Analysis files generated:"
ls -la analysis_*.json 2>/dev/null || echo "   No analysis files yet - uncomment models in script"
echo ""
echo "📋 Next steps:"
echo "   1. Uncomment the models you have in this script"
echo "   2. Or manually run: ollama run MODEL < full_prompt.txt > analysis_MODEL.json"
echo "   3. Then run: python3 aggregate_insights.py"
echo "============================================================"
