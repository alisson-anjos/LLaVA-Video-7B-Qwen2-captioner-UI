import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM

# Configuration parameters
INPUT_CSV = "/home/cseti/Data/Datasets/videos/Arcane/Cut Original/best_of/jinx/16x1360x768/captions.csv"                # Path to input CSV file
OUTPUT_CSV = "/home/cseti/Data/Datasets/videos/Arcane/Cut Original/best_of/jinx/16x1360x768/output.csv"              # Path to save the output CSV file
INPUT_COLUMN = "caption"         # Name of the column containing text to refine
OUTPUT_COLUMN = "refined_text"         # Name of the column where refined text will be saved
OLLAMA_MODEL = "llama3.2:3b"               # Name of the Ollama model to use
MAX_TOKENS = 200                      # Maximum number of tokens for the refined text
BATCH_SIZE = 10                       # Number of rows to process before saving progress

# Prompt templates
SYSTEM_PROMPT = """
You are an AI prompt engineer tasked with helping me modifying a list of automatically generated prompts.

Keep the original text but only do the following modifications:
- you responses should just be the prompt
- do not mention your task or the text itself
- add the following word to the start of each prompt: csetiarcane
- modify each text so that nfjinx is the main character in all of them, so use her name and since she's a woman, refer to her gender when necessary to make the sentences meaningful.
- remove references to video such as "the video begins" or "the video features" etc., but keep those sentences meaningful
- use only declarative sentences
""".strip()

USER_PROMPT = """
Could you enhance and refine the following text while maintaining its core meaning:

```
{0}
```

Please limit the response to [{1}] tokens.
""".strip()

def refine_text(text: str, model: OllamaLLM, max_tokens: int) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT.format(text, max_tokens)),
    ]
    
    try:
        output = model.invoke(messages)  # This returns a string directly
        return output  # No need to access .content
    except Exception as e:
        print(f"Error refining text: {str(e)}")
        return ""

def main():
    # Initialize Ollama model
    model = OllamaLLM(model=OLLAMA_MODEL)
    
    try:
        # Read input CSV
        df = pd.read_csv(INPUT_CSV)
        
        # Create output column if it doesn't exist
        if OUTPUT_COLUMN not in df.columns:
            df[OUTPUT_COLUMN] = ""
        
        # Process rows in batches
        total_rows = len(df)
        for i in range(0, total_rows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_rows)
            print(f"\nProcessing rows {i+1} to {batch_end} of {total_rows}")
            
            for idx in range(i, batch_end):
                if pd.isna(df.loc[idx, INPUT_COLUMN]):
                    print(f"Skipping row {idx+1}: Empty input text")
                    continue
                    
                if df.loc[idx, OUTPUT_COLUMN] != "":
                    print(f"Skipping row {idx+1}: Already processed")
                    continue
                
                print(f"Processing row {idx+1}...")
                input_text = str(df.loc[idx, INPUT_COLUMN])
                refined_text = refine_text(input_text, model, MAX_TOKENS)
                df.loc[idx, OUTPUT_COLUMN] = refined_text
            
            # Save progress after each batch
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"Progress saved to {OUTPUT_CSV}")
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
    
    finally:
        print("\nProcessing complete!")

if __name__ == "__main__":
    main()