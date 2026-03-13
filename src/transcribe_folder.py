import os
import glob
import json
import whisperx
import torch
import gc
from dotenv import load_dotenv

from whisperx.diarize import DiarizationPipeline

# Load environment variables from .env file
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../data/lt")  # Place your audio files here
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../transcriptions/lt")  # Transcriptions will be saved here
HF_TOKEN = os.getenv("HF_TOKEN") # Safely loads from .env

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing! Please add it to your .env file.")

# ... (the rest of the script remains exactly the same)
# RTX 5080 Optimization
DEVICE = "cuda"
COMPUTE_TYPE = "float16" # The 5080 handles float16 natively, saving VRAM and boosting speed
BATCH_SIZE = 16          # You can likely push this to 32 on a 5080 if you want

# This prompt tricks the AI into expecting both languages so it doesn't force translate
INITIAL_PROMPT = "Okay, let's practice. Hola, ¿cómo estás? I am doing well, estoy bien."

# Supported audio formats
AUDIO_EXTENSIONS = ("*.mp3", "*.wav", "*.m4a", "*.flac")

# ==========================================

def process_folder():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Gather all audio files
    audio_files =[]
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not audio_files:
        print(f"No audio files found in {INPUT_FOLDER}.")
        return

    print(f"Found {len(audio_files)} files. Loading models into VRAM...")

    # 1. Load the Whisper model (large-v3 is best for code-switching/multilingual)
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, use_auth_token=HF_TOKEN)
    
    # 2. Load the Diarization (Speaker ID) model
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)

    # Process each file
    for file_path in sorted(audio_files):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        json_output = os.path.join(OUTPUT_FOLDER, f"{base_name}.json")
        txt_output = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
        print(f"\n--- Processing: {filename} ---")

        if os.path.exists(json_output) or os.path.exists(txt_output):
            print(f"  Output for {filename} already exists. Skipping to avoid overwriting.")
            continue

        # Load audio into memory
        audio = whisperx.load_audio(file_path)

        # Transcribe
        print("1/4 Transcribing...")
        # Passing initial_prompt to help with the English/Spanish switching
        result = model.transcribe(audio, batch_size=BATCH_SIZE, language=None, print_progress=True)
        
        # Free VRAM associated with transcription to make room for alignment
        detected_language = result["language"]
        print(f"Detected dominant language: {detected_language}")

        # Align timestamps (makes timestamps exact to the word)
        print("2/4 Aligning audio to words...")
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
        
        # Unload alignment model to free VRAM
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        # Diarize (identify speakers)
        print("3/4 Identifying speakers...")
        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=2)

        # Assign speakers to the aligned words
        print("4/4 Merging speakers with text...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Save Output - JSON (Contains full raw data & timestamps)
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        # Save Output - Human Readable TXT
        with open(txt_output, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                # If pyannote fails to identify a speaker for a micro-segment, default to "UNKNOWN"
                speaker = segment.get("speaker", "UNKNOWN")
                start = round(segment["start"], 2)
                end = round(segment["end"], 2)
                text = segment["text"].strip()
                
                line = f"[{start}s - {end}s] {speaker}: {text}\n"
                f.write(line)

        print(f" Finished {filename}. Saved to {OUTPUT_FOLDER}")

    print("\n All files processed successfully!")

if __name__ == "__main__":
    process_folder()