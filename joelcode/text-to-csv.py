import os
import re
import csv

def clean_transcript(text):
    """
    Extracts dialogue, removes HTML tags, timestamps, 
    and non-speech notes.
    """
    # 1. Strip HTML tags like <p> and </p>
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Remove timestamps like [0:02:47] or [09:23]
    text = re.sub(r'\[\d+:?\d*:?\d*\]', '', text)
    
    # 3. Remove parenthetical notes like (inaudible 0:01:07) or (ph)
    text = re.sub(r'\(.*?\)', '', text)
    
    # 4. Extract only the lines starting with the speaker labels
    # This ignores the 'BEGIN TRANSCRIPT' and header junk
    lines = text.split('\n')
    dialogue_turns = []
    
    for line in lines:
        line = line.strip()
        # Look for lines starting with COUNSELOR: or PATIENT:
        if line.startswith("COUNSELOR:") or line.startswith("PATIENT:"):
            dialogue_turns.append(line)
            
    # Join turns with a space to create a single continuous transcript string
    return " ".join(dialogue_turns)

def convert_directory_to_csv(input_folder, output_csv):
    results = []
    
    # Process every .txt file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                
            clean_text = clean_transcript(raw_content)
            
            if clean_text:
                results.append({
                    "session_id": filename.replace(".txt", ""),
                    "transcript_history": clean_text
                })

    # Save to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "transcript_history"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Successfully converted {len(results)} files to {output_csv}")

# --- Execution ---
# Replace 'transcripts' with your folder name
# convert_directory_to_csv("path/to/your/text/files", "real_therapy_dataset.csv")
