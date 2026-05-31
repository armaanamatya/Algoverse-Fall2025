# Notebook Update Summary

## ✅ Completed Changes

### 1. **Added HTML Transcript Parser** (`transcript_parser.py`)
   - `parse_html_transcript_text()` - Parses HTML-formatted transcripts
   - `parse_html_transcript_file()` - Convenience function for files
   - `parse_transcript_auto()` - Auto-detects format (HTML vs plain text)
   - Handles:
     - `<p>PATIENT: content</p>` and `<p>COUNSELOR: content</p>` tags
     - Timestamps in format `0:MM:SS.S` or `M:MM:SS.S`
     - `(inaudible XX:XX)` markers (removes them)

### 2. **Updated Both Notebooks**

#### `therapy_memincluded.ipynb` (Memory-Enhanced Evaluation)
   - ✅ Imports `parse_html_transcript_file`
   - ✅ Section 4: Loads ALL 16 files from `0518-014_raw/`
   - ✅ Section 5: Processes each file in batch
   - ✅ Uses `evaluate_cbt_adherence_with_memory()` and `evaluate_persona_consistency_with_memory()`
   - ✅ Saves results as `{filename}_eval.json` in `evaluation_results/` directory

#### `therapy_memnotincluded.ipynb` (Standard Evaluation)
   - ✅ Imports `parse_html_transcript_file`
   - ✅ Section 4: Loads ALL 16 files from `0518-014_raw/`
   - ✅ Section 5: Processes each file in batch
   - ✅ Uses `evaluate_cbt_adherence()` and `evaluate_persona_consistency()` (no memory passed to evaluators)
   - ✅ Saves results as `{filename}_eval.json` in `evaluation_results/` directory

### 3. **Output Format**

Each file will generate a JSON output with this structure:
```json
{
  "filename": "1000056544.txt",
  "total_turns": 250,
  "counselor_turns_evaluated": 125,
  "patient_turns_processed": 125,
  "model": "gpt-oss:20b",
  "cbt_adherence_results": [...],
  "persona_consistency_results": [...],
  "memory_snapshots": [...]
}
```

## 📁 Files Modified

1. `our-pipeline/transcript_parser.py` - Added HTML parsing functions
2. `therapy_memincluded.ipynb` - Updated for batch processing with memory
3. `therapy_memnotincluded.ipynb` - Updated for batch processing without memory

## 🚀 How to Run

1. **Start Jupyter Notebook:**
   ```powershell
   jupyter notebook
   ```

2. **Open either notebook:**
   - `therapy_memincluded.ipynb` - Evaluators receive memory context
   - `therapy_memnotincluded.ipynb` - Evaluators do NOT receive memory context

3. **Run all cells** - The notebook will:
   - Load all 16 transcript files from `0518-014_raw/`
   - Parse each using the HTML parser
   - Evaluate each file with gpt-oss:20b
   - Save individual JSON results to `evaluation_results/`

4. **Results will be saved as:**
   - `evaluation_results/1000056544_eval.json`
   - `evaluation_results/1000056545_eval.json`
   - ... (one for each of the 16 files)

## ⚙️ Configuration

- **Model**: Already set to `gpt-oss:20b` in both notebooks
- **MAX_TURNS**: Set to `None` (processes all turns). Change to a number (e.g., `20`) for testing
- **DELAY_BETWEEN_CALLS**: Set to `0.1` seconds for Ollama (fast local inference)

## 📊 Dataset Info

- **Location**: `0518-014_raw/`
- **Files**: 16 therapy transcript files (`.txt`)
- **Format**: HTML-tagged with `<p>COUNSELOR:</p>` and `<p>PATIENT:</p>` markers
- **Example file**: `1000056544.txt` (250 turns, 125 counselor + 125 patient)

## ✨ Key Features

1. **Batch Processing**: All 16 files processed automatically
2. **Individual Outputs**: Each file gets its own `_eval.json` file
3. **Progress Tracking**: Shows which file is being processed and progress every 10 turns
4. **Proper Parsing**: Correctly extracts counselor and patient turns from HTML format
5. **Memory Comparison**: Two notebooks allow comparing results with/without memory context in evaluation
