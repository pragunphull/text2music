# evaluation_utils.py
import pretty_midi
import numpy as np
from collections import Counter

def pitch_histogram_similarity(midi1: pretty_midi.PrettyMIDI, midi2: pretty_midi.PrettyMIDI):
    """
    Calculates the similarity between the pitch histograms of two MIDI files.
    """
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

if __name__ == '__main__':
    # Example Usage (replace with your MIDI file paths)
    midi_file1 = pretty_midi.PrettyMIDI('path/to/your/midi1.mid')  # Replace with actual paths
    midi_file2 = pretty_midi.PrettyMIDI('path/to/your/midi2.mid')

    pitch_sim = pitch_histogram_similarity(midi_file1, midi_file2)
    print(f"Pitch Histogram Similarity: {pitch_sim}")

    rhythm_sim_value = rhythm_similarity(midi_file1, midi_file2)
    print(f"Rhythm Similarity: {rhythm_sim_value}")

    precision, recall_value = note_precision_recall(midi_file1, midi_file2)
    print(f"Note Precision: {precision}, Recall: {recall_value}")