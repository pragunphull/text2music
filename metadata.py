import os
import json
from mido import MidiFile

def extract_tempo(midi_file_path):

    try:
        midi_file = MidiFile(midi_file_path)
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return msg.tempo
        return None  # Return None if no tempo message is found
    except Exception as e:
        print(f"Error processing {midi_file_path}: {e}")
        return None

def update_json_with_tempo_from_dir(midi_dir_path, json_file_path):

    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    updated_data = []
    for item in json_data:
        midi_filename_base = item.get('filename')  # Adjust 'filename' to your key
        if midi_filename_base:
            midi_file_path = os.path.join(midi_dir_path, f"{midi_filename_base}.mid") # Assuming .mid extension
            if os.path.exists(midi_file_path):
                tempo = extract_tempo(midi_file_path)
                if tempo is not None:
                    item['tempo'] = tempo
                else:
                    print(f"Warning: No tempo information found in {midi_file_path}")
            else:
                print(f"Warning: MIDI file not found for {midi_filename_base} in the specified directory.")
        updated_data.append(item)

    with open(json_file_path, 'w') as f:
        json.dump(updated_data, f, indent=4)

# Specify your file locations (using raw strings)
midi_directory = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\dataset midi"
json_file = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\maestro-v3.0.0.json"

update_json_with_tempo_from_dir(midi_directory, json_file)

print("Tempo metadata extraction and JSON update from directory complete.")