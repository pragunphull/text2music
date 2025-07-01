import os
import shutil

def move_midi_files(source_folder, destination_folder):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(".midi"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                try:
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {file}")
                except Exception as e:
                    print(f"Error moving {file}: {e}")


source_folder = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\maestro-v3.0.0"  
destination_folder = r"C:\Users\pragun phull\OneDrive\Desktop\text-to-music\dataset midi"

move_midi_files(source_folder, destination_folder)

print("MIDI file consolidation complete.")