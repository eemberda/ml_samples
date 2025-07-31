import os
import sys
# import speech_recognition as sr # No longer needed for Whisper
import whisper # Import the whisper library
# import pydub # Still useful if you need to handle different audio formats or check duration

def transcribe_mp3_to_text_whisper(mp3_file_path, model_name="base"):
    """
    Transcribes an MP3 audio file into text using the Whisper ASR model.

    Args:
        mp3_file_path (str): The path to the MP3 audio file.
        model_name (str): The name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
                          Larger models are more accurate but slower and require more memory.

    Returns:
        str: The transcribed text, or None if an error occurred.
    """
    if not os.path.exists(mp3_file_path):
        print(f"Error: MP3 file not found at '{mp3_file_path}'")
        return None

    try:
        print(f"Loading Whisper model '{model_name}' (this may take some time the first time)...")
        model = whisper.load_model(model_name)
        print("Whisper model loaded.")

        print(f"Transcribing audio file: {mp3_file_path} using Whisper...")
        # Whisper can directly process various audio formats, including MP3.
        # It handles the internal audio loading and processing.
        result = model.transcribe(mp3_file_path)
        
        return result["text"]

    except Exception as e:
        print(f"An error occurred during Whisper transcription: {e}")
        print("Please ensure you have whisper installed: pip install openai-whisper")
        print("Also, ensure you have FFmpeg installed on your system and added to your PATH.")
        return None


if __name__ == "__main__":
    # IMPORTANT: Install necessary libraries:
    # pip install openai-whisper
    # pip install pydub (still good to have for general audio handling, though whisper handles MP3 directly)
    # Ensure you have FFmpeg installed on your system and added to your PATH
    # (download from https://ffmpeg.org/download.html)
    # For faster inference with GPU, you might need:
    # pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 (for CUDA 11.8, adjust as needed)

    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_mp3_file> [whisper_model_name]")
        print("Example: python transcribe_audio_whisper.py audio.mp3")
        print("Example: python transcribe_audio_whisper.py long_meeting.mp3 medium")
        sys.exit(1)

    mp3_file_path = sys.argv[1] 
    
    # Optional: Allow specifying the Whisper model name as a second argument
    whisper_model = "base" # Default model
    if len(sys.argv) >= 3:
        user_model = sys.argv[2].lower()
        supported_models = ["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en"]
        if user_model in supported_models:
            whisper_model = user_model
        else:
            print(f"Warning: '{user_model}' is not a recognized Whisper model. Using default 'base'.")
            print(f"Supported models: {', '.join(supported_models)}")


    print(f"\nAttempting to transcribe: {mp3_file_path} using Whisper model: {whisper_model}")
    transcribed_text = transcribe_mp3_to_text_whisper(mp3_file_path, model_name=whisper_model)

    if transcribed_text:
        print("\n--- Transcription Result (Whisper) ---")
        print(transcribed_text)
    else:
        print("\nTranscription failed.")

    print("\n--- Instructions ---")
    print("To use this code:")
    print("1. Save the code above as a Python file (e.g., `transcribe_audio_whisper.py`).")
    print("2. Open your terminal or command prompt.")
    print("3. Navigate to the directory where you saved the file.")
    print("4. Install the whisper library:")
    print("   pip install openai-whisper")
    print("   (Optional: pip install torch torchaudio for GPU acceleration if available)")
    print("5. Make sure FFmpeg is installed on your system and accessible via your PATH.")
    print("   (Download from https://ffmpeg.org/download.html if needed)")
    print("6. Run the script with your MP3 file path as an argument:")
    print("   python transcribe_audio_whisper.py your_audio_file.mp3")
    print("   You can optionally specify a model (e.g., 'small', 'medium', 'large'):")
    print("   python transcribe_audio_whisper.py your_audio_file.mp3 medium")