import torch
import torch.nn.functional as F
import numpy as np
import re
from postprocessing import inverse_scale_piano_roll, piano_roll_to_midi # Ensure postprocessing.py is accessible

# --- Configuration (Adjust to your setup) ---
# Load necessary parameters from your training or preprocessing
processed_data = torch.load('maestro_text_to_pianoroll_processed.pt') # Load saved data
feature_metadata = processed_data['feature_metadata']
tfidf = feature_metadata['tfidf']
encoder = feature_metadata['encoder']
scaler = feature_metadata['scaler']
numerical_features = feature_metadata['numerical_features']
categorical_features = feature_metadata['categorical_features']

text_vocab_size = ... # Determine from your tokenizer or embedding layer
structured_feature_dim = feature_metadata['structured_features'].shape[1]  # Get from saved data
max_seq_len = 20  # Or load from saved metadata if you saved it
num_patches = 400 # Or load
num_pitches = 128
embedding_dim = 256 # Or load
nhead = 8 # Or load
num_encoder_layers = 4 # Or load
num_decoder_layers = 4 # Or load
dropout = 0.1 # Or load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = PianoRollTransformer(
    text_vocab_size=text_vocab_size,
    structured_feature_dim=structured_feature_dim,
    max_seq_len=max_seq_len,
    num_patches=num_patches,
    num_pitches=num_pitches,
    embedding_dim=embedding_dim,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dropout=dropout
).to(device)
model.load_state_dict(torch.load('best_model.pth')) # Load your trained weights
model.eval()

# --- Input Text and Processing ---
input_text = "mozart sonata A minor 100 tempo"

# 1. Text Embedding
cleaned_text = re.sub(r'[^\\w\\s]', '', input_text.lower()).strip()
text_embedding = tfidf.transform([cleaned_text]).toarray()

# 2. Structured Features
# Initialize structured features (important to match training data structure)
structured_features_input = np.zeros((1, structured_feature_dim)) # Start with zeros

# Extract key (minor/major)
key = "minor" if "minor" in input_text.lower() else "major"
key_index = 0 # Index within categorical features (adjust if needed)
key_col_name = f"canonical_composer_{key}" # Construct column name
try:
    key_col_index = np.where(feature_metadata['categorical_columns'] == key_col_name)[0][0]
    structured_features_input[0, numerical_features.shape[1] + key_col_index] = 1 # Offset by numerical feature count
except IndexError:
    print(f"Warning: Key '{key}' not found in training data's categorical features.")

# Extract tempo
tempo_match = re.search(r'(\\d+)\\s*tempo', input_text.lower())
tempo = int(tempo_match.group(1)) if tempo_match else 120 # Default tempo
tempo_scaled = scaler.transform(np.array([[tempo]]))[0, 0] # Scale the tempo
structured_features_input[0, 0] = tempo_scaled # Assuming tempo is the first numerical feature


# Convert to tensors
text_tensor = torch.tensor(text_embedding, dtype=torch.float32).to(device)
structured_tensor = torch.tensor(structured_features_input, dtype=torch.float32).to(device)


# --- Generation ---
max_generation_length = 400 # Adjust as needed
start_pitch_token = 60.0 # Adjust based on your data

def generate_piano_roll(model, src_text, structured_features, max_length, start_pitch, device):
    model.eval()
    with torch.no_grad():
        encoder_output = model.transformer_encoder(model.positional_encoding(model.text_embedding(src_text) + model.structured_embedding(structured_features)))
        tgt_sequence = torch.tensor([[start_pitch, 0.5]]).unsqueeze(0).to(device) # [start_pitch, initial velocity]

        for _ in range(max_length):
            time_steps = torch.arange(tgt_sequence.size(1), device=device).unsqueeze(0)
            tgt_embedded = model.positional_encoding(model.pitch_embedding(tgt_sequence[:, :, 0].long()) + model.time_embedding(time_steps))
            decoder_output = model.transformer_decoder(tgt_embedded, encoder_output)
            pitch_logits = model.fc_pitch(decoder_output[:, -1, :])
            velocity_prediction = torch.sigmoid(model.fc_velocity(decoder_output[:, -1, :]))

            next_pitch = torch.argmax(pitch_logits, dim=-1).unsqueeze(-1).float()
            next_velocity = velocity_prediction.squeeze(-1).unsqueeze(-1)

            next_token = torch.cat([next_pitch, next_velocity], dim=-1).unsqueeze(0)
            tgt_sequence = torch.cat([tgt_sequence, next_token], dim=1)

        return tgt_sequence.squeeze(0).cpu().numpy()

generated_piano_roll = generate_piano_roll(model, text_tensor, structured_tensor, max_generation_length, start_pitch_token, device)

# --- Post-processing and MIDI Conversion ---
# Inverse scaling (if applied during training)
generated_piano_roll_original_scale = inverse_scale_piano_roll(generated_piano_roll, scaler)

# Convert to MIDI
midi_output = piano_roll_to_midi(generated_piano_roll_original_scale, fs=100)  # Adjust fs as needed
midi_output.write('generated_mozart_sonata.mid')

print("MIDI file 'generated_mozart_sonata.mid' created.")