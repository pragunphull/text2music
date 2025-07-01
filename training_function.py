import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pretty_midi  # Make sure you have this installed
from collections import Counter # Make sure you have this installed

# --- Evaluation Utility Functions (as provided) ---
def pitch_histogram_similarity(midi1: pretty_midi.PrettyMIDI, midi2: pretty_midi.PrettyMIDI):
    """Calculates the similarity between the pitch histograms of two MIDI files."""
    hist1 = [0] * 12  # Chromatic scale
    hist2 = [0] * 12

    for instrument in midi1.instruments:
        for note in instrument.notes:
            pitch_class = note.pitch % 12
            hist1[pitch_class] += 1

    for instrument in midi2.instruments:
        for note in instrument.notes:
            pitch_class = note.pitch % 12
            hist2[pitch_class] += 1

    # Normalize histograms
    sum_hist1 = sum(hist1)
    if sum_hist1 > 0:
        hist1 = [x / sum_hist1 for x in hist1]

    sum_hist2 = sum(hist2)
    if sum_hist2 > 0:
        hist2 = [x / sum_hist2 for x in hist2]

    # Calculate cosine similarity
    dot_product = sum(p1 * p2 for p1, p2 in zip(hist1, hist2))
    magnitude1 = sum(p ** 2 for p in hist1) ** 0.5
    magnitude2 = sum(p ** 2 for p in hist2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Or handle this case as you see fit
    return dot_product / (magnitude1 * magnitude2)

def rhythm_similarity(midi1: pretty_midi.PrettyMIDI, midi2: pretty_midi.PrettyMIDI):
    """
    Calculates a simplified rhythm similarity based on note durations.
    This is a placeholder and needs a more sophisticated approach.
    """
    durations1 = []
    for instrument in midi1.instruments:
        for note in instrument.notes:
            durations1.append(note.end - note.start)

    durations2 = []
    for instrument in midi2.instruments:
        for note in instrument.notes:
            durations2.append(note.end - note.start)

    if not durations1 or not durations2:
        return 0  # Handle empty cases

    # Very basic similarity (average duration difference) - Replace this!
    avg_diff = np.mean(np.abs(np.mean(durations1) - np.mean(durations2)))
    return avg_diff

def note_precision_recall(generated_midi: pretty_midi.PrettyMIDI, ground_truth_midi: pretty_midi.PrettyMIDI):
    """
    Calculates note-level precision and recall.
    """
    generated_notes = []
    for instrument in generated_midi.instruments:
        generated_notes.extend([(note.start, note.pitch) for note in instrument.notes])

    ground_truth_notes = []
    for instrument in ground_truth_midi.instruments:
        ground_truth_notes.extend([(note.start, note.pitch) for note in instrument.notes])

    if not generated_notes and not ground_truth_notes:
        return 1.0, 1.0  # Both empty, perfect match
    if not generated_notes or not ground_truth_notes:
        return 0.0, 0.0  # One is empty, no match

    correct_notes = sum(1 for note in generated_notes if note in ground_truth_notes)

    precision = correct_notes / len(generated_notes) if generated_notes else 0.0
    recall = correct_notes / len(ground_truth_notes) if ground_truth_notes else 0.0

    return precision, recall

# --- Modified training_function.py ---
def pianoroll_to_pretty_midi(pianoroll, fs=100): 


    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  

    batch_size, seq_len, _ = pianoroll.shape
    for b in range(batch_size):
        current_time = 0
        for t in range(seq_len):
            pitch = int(pianoroll[b, t, 0].item())
            velocity = int(pianoroll[b, t, 1].item() * 127)  # Scale velocity to 0-127

            if velocity > 0:  
                start_time = current_time / fs
                end_time = (current_time + 1) / fs  

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
            current_time += 1

        midi.instruments.append(instrument)
    return midi

def train_model(processed_data_path, model, optimizer, criterion_pitch, criterion_velocity, batch_size, num_epochs, device):
    # Load the processed data
    data = torch.load(processed_data_path)
    text_sequences = data['text_sequences']
    structured_features = data['structured_features']
    pianorolls = data['pianorolls']

    # Split data into training and validation sets
    train_text, val_text, train_structured, val_structured, train_pianorolls, val_pianorolls = train_test_split(
        text_sequences, structured_features, pianorolls, test_size=0.1, random_state=42
    )

    train_dataset = TensorDataset(train_text, train_structured, train_pianorolls)
    val_dataset = TensorDataset(val_text, val_structured, val_pianorolls)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (src_text, structured, tgt_pianoroll) in enumerate(train_loader):
            src_text = src_text.to(device)
            structured = structured.to(device)
            tgt_pianoroll = tgt_pianoroll.to(device)

            # Teacher forcing: use the target as input for the decoder
            # Shift the target piano roll by one time step
            tgt_input = tgt_pianoroll[:, :-1, :]
            tgt_output_pitch = tgt_pianoroll[:, 1:, 0].long()
            tgt_output_velocity = tgt_pianoroll[:, 1:, 1].unsqueeze(-1)

            optimizer.zero_grad()
            pitch_predictions, velocity_predictions = model(src_text, structured, tgt_input)

            # Calculate loss
            pitch_loss = criterion_pitch(pitch_predictions.view(-1, model.num_pitches + 1), tgt_output_pitch.view(-1))
            velocity_loss = criterion_velocity(velocity_predictions, tgt_output_velocity)
            loss = pitch_loss + velocity_loss

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}')

        # Validation step
        model.eval()
        total_val_loss = 0
        all_generated_midis = []
        all_target_midis = []

        with torch.no_grad():
            for src_text_val, structured_val, tgt_pianoroll_val in val_loader:
                src_text_val = src_text_val.to(device)
                structured_val = structured_val.to(device)
                tgt_pianoroll_val = tgt_pianoroll_val.to(device)

                tgt_input_val = tgt_pianoroll_val[:, :-1, :]
                tgt_output_pitch_val = tgt_pianoroll_val[:, 1:, 0].long()
                tgt_output_velocity_val = tgt_pianoroll_val[:, 1:, 1].unsqueeze(-1)

                pitch_predictions_val, velocity_predictions_val = model(src_text_val, structured_val, tgt_input_val)
                pitch_loss_val = criterion_pitch(pitch_predictions_val.view(-1, model.num_pitches + 1), tgt_output_pitch_val.view(-1))
                velocity_loss_val = criterion_velocity(velocity_predictions_val, tgt_output_velocity_val)
                val_loss = pitch_loss_val + velocity_loss_val
                total_val_loss += val_loss.item()

                # Convert predicted and target piano rolls to PrettyMIDI objects
                generated_pianoroll = torch.stack((torch.argmax(pitch_predictions_val, dim=-1).float(), velocity_predictions_val.squeeze(-1)), dim=-1)
                
                generated_midi = pianoroll_to_pretty_midi(generated_pianoroll.cpu())
                target_midi = pianoroll_to_pretty_midi(tgt_pianoroll_val.cpu())

                all_generated_midis.append(generated_midi)
                all_target_midis.append(target_midi)

        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}')

        # Calculate and print evaluation metrics
        total_pitch_sim = 0
        total_rhythm_sim = 0
        total_precision = 0
        total_recall = 0

        for gen_midi, target_midi in zip(all_generated_midis, all_target_midis):
            total_pitch_sim += pitch_histogram_similarity(gen_midi, target_midi)
            total_rhythm_sim += rhythm_similarity(gen_midi, target_midi)
            precision, recall = note_precision_recall(gen_midi, target_midi)
            total_precision += precision
            total_recall += recall

        num_samples = len(all_generated_midis)
        if num_samples > 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Pitch Sim: {total_pitch_sim / num_samples:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Rhythm Sim: {total_rhythm_sim / num_samples:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Precision: {total_precision / num_samples:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Recall: {total_recall / num_samples:.4f}')

        # Save the best model weights (you might want to base this on a combination of metrics)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Validation loss improved. Saving model weights to best_model.pth')

    print('Training finished.')