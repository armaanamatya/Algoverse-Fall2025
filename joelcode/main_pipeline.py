import pandas as pd
import os
from openai import OpenAI
from nemo_curator.services import OpenAIClient
from nemo_curator.synthetic import NemotronGenerator

# --- Configuration ---
WINDOW_SIZE = 10      # How many of the most recent turns to keep in context
TOTAL_EXTRA_TURNS = 100 # Adjust to 1000+ for your actual research
CHUNK_SIZE = 4         # How many turns to generate before shifting the window
SEED_CSV = "real_therapy_dataset.csv"

# --- Setup ---
api_key = os.getenv("NIM_API_KEY")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
llm_client = OpenAIClient(client)
generator = NemotronGenerator(llm_client)

def get_sliding_window(full_list, window_size):
    """Slices the last N items from a list."""
    return full_list[-window_size:] if len(full_list) > window_size else full_list

def run_sliding_window_session(initial_transcript):
    # Start with your real seed data
    # We treat the seed as the initial 'full_conversation'
    full_conversation = [initial_transcript]
    
    for _ in range(0, TOTAL_EXTRA_TURNS, CHUNK_SIZE):
        # 1. Prepare the window (The 'Memory' of the LLM)
        # We only pass the last WINDOW_SIZE turns to the generator
        window_context = get_sliding_window(full_conversation, WINDOW_SIZE)
        context_string = "\n".join(window_context)

        # 2. Generate next chunk
        # Note: We use the history parameter to feed the window
        new_turns = generator.generate_dialogue(
            openline=None, 
            user_model="meta/llama-3.1-70b-instruct",
            assistant_model="meta/llama-3.3-70b-instruct",
            n_user_turns=CHUNK_SIZE // 2,
            user_prompt_template="You are the Patient. Use this recent history to continue: {history}",
            assistant_prompt_template="You are the Counselor. Use this recent history to continue: {history}",
            history=context_string
        )
        
        # 3. Append to the total log
        # Each item in 'new_turns' is a dict: {'role': 'user/assistant', 'content': '...'}
        formatted_turns = [f"{t['role'].capitalize()}: {t['content']}" for t in new_turns]
        full_conversation.extend(formatted_turns)
        
    return full_conversation

# --- Execution ---
df = pd.read_csv(SEED_CSV)
results = []

for idx, row in df.iterrows():
    print(f"Processing Session {row['session_id']}...")
    long_session = run_sliding_window_session(row['transcript_history'])
    
    results.append({
        "session_id": row['session_id'],
        "full_synthetic_dialogue": long_session
    })

# Save output
pd.DataFrame(results).to_json("sliding_window_research_data.jsonl", orient="records", lines=True)
print("Generation Complete.")
