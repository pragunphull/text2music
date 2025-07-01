import os
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def midi_to_piano_roll(midi_file, fs=100):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return piano_roll

def standard_scale_piano_roll(piano_roll):
    """
    Standard scales a single piano roll.  Handles edge cases.

    Args:
        piano_roll (np.ndarray): A 2D numpy array representing the piano roll.

    Returns:
        np.ndarray: The scaled piano roll, or the original if scaling isn't possible.
        StandardScaler: The scaler used, or None if no scaling was done.
    """
    if piano_roll.size <= 1:  # Nothing to scale if the array is empty or has only one element
        return piano_roll, None  # Return original and None scaler

    # Reshape for scaling
    original_shape = piano_roll.shape
    reshaped_roll = piano_roll.reshape(-1, original_shape[-1])

    # Apply scaling
    scaler = StandardScaler()
    try:
        scaled_data = scaler.fit_transform(reshaped_roll)
    except ValueError: # occurs when the input has zero variance
        return piano_roll, None

    # Reshape back to the original shape
    scaled_piano_roll = scaled_data.reshape(original_shape)
    return scaled_piano_roll, scaler


midi_dir = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\dataset midi"
output_dir = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\piano_rolls"


os.makedirs(output_dir, exist_ok=True)


scalers = {} # Dictionary to store scalers, keys are filenames

for midi_file in os.listdir(midi_dir):
    if midi_file.endswith(".mid") or midi_file.endswith(".midi"):
        midi_path = os.path.join(midi_dir, midi_file)
        try:
            piano_roll = midi_to_piano_roll(midi_path)
            
            #scale piano roll
            scaled_piano_roll, scaler = standard_scale_piano_roll(piano_roll)
            
            # Save scaled piano roll directly to output_dir
            output_file = os.path.join(output_dir, midi_file.replace(".mid", ".npy").replace(".midi", ".npy"))
            np.save(output_file, scaled_piano_roll)
            
            if scaler: # only save scaler if it is not None
                scalers[midi_file] = scaler #store scaler
            
            print(f"Processed {midi_file}: Piano Roll Shape = {piano_roll.shape}, Scaled Piano Roll Shape = {scaled_piano_roll.shape}")
            

        except Exception as e:
            print(f"Error processing {midi_file}: {e}")

# Save the scalers dictionary.  This is crucial for being able to invert the scaling later.
np.save(os.path.join(output_dir, "scalers.npy"), scalers)
print(f"Saved scalers to {os.path.join(output_dir, 'scalers.npy')}")