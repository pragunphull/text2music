import os
import numpy as np
import pretty_midi

# Input and output paths
input_folder = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\generated_piano_rolls"
output_folder = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\generated_music"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

def piano_roll_to_midi(piano_roll, fs=100, program=0):
    """Convert a piano roll array to a PrettyMIDI object."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    piano_roll = (piano_roll > 0).astype(np.uint8)  # Binarize if needed

    # Create note events
    for note in range(piano_roll.shape[0]):
        note_on = None
        for t in range(piano_roll.shape[1]):
            if piano_roll[note, t] and note_on is None:
                note_on = t
            elif not piano_roll[note, t] and note_on is not None:
                start = note_on / fs
                end = t / fs
                instrument.notes.append(pretty_midi.Note(velocity=100, pitch=note, start=start, end=end))
                note_on = None
        if note_on is not None:  # Sustain note till end if never turned off
            start = note_on / fs
            end = piano_roll.shape[1] / fs
            instrument.notes.append(pretty_midi.Note(velocity=100, pitch=note, start=start, end=end))

    midi.instruments.append(instrument)
    return midi

# Process each .npy file
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        filepath = os.path.join(input_folder, filename)
        piano_roll = np.load(filepath)
        
        if len(piano_roll.shape) == 3:
            piano_roll = piano_roll[0]  # If shape is (1, 128, time), squeeze it
        
        midi = piano_roll_to_midi(piano_roll)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mid')
        midi.write(output_path)
        print(f"Saved: {output_path}")
