import os
import sys
import time
import signal
import sounddevice as sd
import numpy as np
import soundfile as sf
import keyboard
from openai import OpenAI
import pyttsx3
import json
import subprocess
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Assistant Configuration
NOT_DANGEROUS = os.getenv("NOT_DANGEROUS", "False").lower() == "true"
TALKBACK = os.getenv("TALKBACK", "True").lower() == "true"

# Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large")

# TTS Configuration
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai")

# Input/Output Configuration
INPUT_MODE = os.getenv("INPUT_MODE", "voice")
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "both")

# Audio Recording Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORDING_DURATION = 45  # seconds

# Control flags
running = True
recording = False

def signal_handler(sig, frame):
    """Handle interrupt signals for graceful shutdown."""
    global running
    print("\nInterrupt received. Shutting down...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def toggle_recording():
    """Toggle the recording state."""
    global recording
    recording = not recording
    if recording:
        print("Recording started. Press 'V' again to stop or wait 45 seconds.")
    else:
        print("Recording stopped.")

def record_audio():
    """Record audio input from the user."""
    global recording
    audio_data = []
    start_time = time.time()
    
    while recording and time.time() - start_time < MAX_RECORDING_DURATION:
        frame = sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()
        audio_data.append(frame)
    
    recording = False
    print("Recording finished.")
    return np.concatenate(audio_data, axis=0)

def transcribe_audio(audio_data):
    """Transcribe the recorded audio data to text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio_data, SAMPLE_RATE)
        temp_audio_file_path = temp_audio_file.name

    try:
        with open(temp_audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file
            )
        os.unlink(temp_audio_file_path)
        return transcription.text.strip()
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        os.unlink(temp_audio_file_path)
        return ""

def generate_command_sequence(task_description):
    """Generate a sequence of commands to accomplish the given task using GPT-4."""
    system_prompt = """
    You are an AI assistant that generates sequences of shell commands to accomplish tasks.
    For each task, provide a list of commands that will achieve the goal.
    Each command should be a dictionary with the following keys:
    - 'command': The actual shell command to be executed.
    - 'description': A brief explanation of what the command does.
    - 'is_dangerous': A boolean indicating if the command is potentially dangerous.
    
    Ensure the commands are correct, efficient, and safe to execute.
    If a task cannot be accomplished safely or requires user interaction, 
    explain why and suggest alternatives if possible.
    
    Respond with a valid JSON array of command dictionaries.
    """
    
    user_prompt = f"Generate a sequence of shell commands to accomplish the following task: {task_description}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Attempt to parse the content as JSON
        try:
            command_sequence = json.loads(content)
            if not isinstance(command_sequence, list):
                raise ValueError("Response is not a list of commands")
            return command_sequence
        except json.JSONDecodeError:
            print(f"Error: Unable to parse GPT-4 response as JSON. Raw response:\n{content}")
            return []
        
    except Exception as e:
        print(f"Error generating command sequence: {str(e)}")
        return []

def execute_command_sequence(command_sequence):
    """Execute a sequence of shell commands and return the results."""
    results = []
    for cmd_info in command_sequence:
        if cmd_info['is_dangerous'] and NOT_DANGEROUS:
            results.append({
                'command': cmd_info['command'],
                'output': 'Command execution skipped due to NOT_DANGEROUS setting.',
                'error': None
            })
            continue
        
        if cmd_info['is_dangerous']:
            print(f"Warning: The following command is potentially dangerous:")
            print(f"  {cmd_info['command']}")
            print(f"Description: {cmd_info['description']}")
            confirmation = input("Do you want to proceed? (yes/no): ").lower()
            if confirmation != 'yes':
                results.append({
                    'command': cmd_info['command'],
                    'output': 'Command execution aborted by user.',
                    'error': None
                })
                continue
        
        try:
            result = subprocess.run(cmd_info['command'], shell=True, capture_output=True, text=True, timeout=60)
            results.append({
                'command': cmd_info['command'],
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            })
        except subprocess.TimeoutExpired:
            results.append({
                'command': cmd_info['command'],
                'output': None,
                'error': 'Command execution timed out after 60 seconds.'
            })
        except Exception as e:
            results.append({
                'command': cmd_info['command'],
                'output': None,
                'error': str(e)
            })
    
    return results

def summarize_results(results):
    """Generate a summary of the command execution results."""
    summary = []
    for result in results:
        if result['error']:
            summary.append(f"Command '{result['command']}' failed: {result['error']}")
        else:
            output_preview = result['output'][:100] + '...' if result['output'] and len(result['output']) > 100 else result['output']
            summary.append(f"Command '{result['command']}' executed successfully. Output: {output_preview}")
    
    return "\n".join(summary)

def speak(text):
    """Convert text to speech and play it."""
    if TTS_PROVIDER == 'openai':
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice=TTS_VOICE,
                input=text
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
                for chunk in response.iter_bytes():
                    temp_audio_file.write(chunk)
                temp_audio_file.close()
                os.system(f"afplay {temp_audio_file.name}")  # For macOS
                # Use 'start' for Windows or 'xdg-open' for Linux
            os.unlink(temp_audio_file.name)
        except Exception as e:
            print(f"Error in OpenAI TTS: {e}")
    else:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error in pyttsx3 TTS: {e}")

def output_response(text):
    """Output the response based on the OUTPUT_MODE setting."""
    if OUTPUT_MODE in ['text', 'both']:
        print(text)
    if OUTPUT_MODE in ['voice', 'both'] and TALKBACK:
        speak(text)

def main():
    """Main function to run the voice/text-controlled assistant."""
    global running, recording
    
    print("Advanced Voice/Text Assistant")
    print(f"Input mode: {INPUT_MODE}")
    print(f"Output mode: {OUTPUT_MODE}")
    if INPUT_MODE == 'voice':
        print("Press 'V' to start/stop recording (max 45 seconds).")
    print("Press 'Ctrl+C' to exit.")
    
    if INPUT_MODE == 'voice':
        keyboard.on_press_key('v', lambda _: toggle_recording())
    
    while running:
        try:
            if INPUT_MODE == 'voice':
                if recording:
                    audio_data = record_audio()
                    if len(audio_data) > 0:
                        command_text = transcribe_audio(audio_data)
                        if command_text:
                            output_response(f"You said: {command_text}")
                        else:
                            output_response("Sorry, I couldn't understand that. Please try again.")
                            continue
                    else:
                        continue
                else:
                    time.sleep(0.1)
                    continue
            else:  # text input mode
                command_text = input("Enter your command: ")
            
            if command_text:
                command_sequence = generate_command_sequence(command_text)
                if not command_sequence:
                    output_response("I'm sorry, I couldn't generate a command sequence for that task. Please try rephrasing your request.")
                    continue
                
                summary = "Here's what I'm going to do:\n"
                for cmd in command_sequence:
                    summary += f"- {cmd['description']}\n"
                
                output_response(summary)
                
                confirmation = input("Do you want me to proceed? (yes/no): ").lower()
                if confirmation == 'yes':
                    results = execute_command_sequence(command_sequence)
                    result_summary = summarize_results(results)
                    output_response("Task completed. Here's a summary of what I did:")
                    output_response(result_summary)
                else:
                    output_response("Task cancelled.")
            
            if INPUT_MODE == 'voice':
                print("Press 'V' to start a new recording.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            output_response("I'm sorry, an error occurred while processing your request.")

if __name__ == "__main__":
    main()
