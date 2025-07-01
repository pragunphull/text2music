# text2music
A transformer based architecture that converts text to music where NLP descriptions are converted into WAV files


This project implements a deep learning pipeline to generate piano roll music from descriptive text inputs, leveraging a Transformer architecture. It processes textual descriptions, extracts structured musical features (like tempo), converts MIDI files to piano rolls, trains a sequence-to-sequence Transformer model, and finally generates new musical pieces in MIDI format based on new text prompts.

Table of Contents
Project Overview

Features

File Structure

Data Preparation

Dataset Acquisition

MIDI File Consolidation

MIDI to Piano Roll Conversion

Metadata Augmentation (Tempo)

Text and Feature Vectorization

Model Training

Music Generation (Inference)

Customization and Further Development

Acknowledgements

##
1. Project Overview
The core idea is to translate human language descriptions (e.g., "a happy melody in C major, fast tempo") into musical piano rolls. This is achieved using a Transformer model that learns to map vectorized text and structured features to a sequence of piano roll representations. The project uses the MAESTRO dataset for training and provides a complete pipeline from data preprocessing to music generation.


2. Features
MIDI File Consolidation: Automatically collects MIDI files from nested subdirectories into a single dataset folder.

MIDI to Piano Roll Conversion: Converts raw MIDI files into numerical piano roll representations, suitable for deep learning models. It includes optional standard scaling for piano roll data.

Tempo Extraction: Extracts tempo information directly from MIDI files and updates JSON metadata.

Text and Structured Feature Vectorization: Utilizes TF-IDF for text embedding and StandardScaler with OneHotEncoder for numerical and categorical metadata features (composer, title, tempo).

Data Pairing and Splitting: Pairs text/structured features with corresponding piano rolls and splits the dataset into training, validation, and test sets.

Transformer Model: Implements a custom Encoder-Decoder Transformer architecture (PianoRollTransformer) capable of handling text, structured features, and generating sequential piano roll data.

Model Training: Provides a training loop with validation, loss calculation (CrossEntropy for pitch, MSE for velocity), and best model saving.

Music Generation: Generates new piano rolls from unseen text prompts using the trained Transformer model and converts them into playable MIDI files.

Inverse Scaling: Reverts the scaling applied to piano roll data during preprocessing, ensuring MIDI output has correct velocity values.


3. File Structure
.
├── dataset midi/                 # Consolidated MIDI files will be stored here
├── maestro-v3.0.0.csv            # MAESTRO dataset metadata (download separately)
├── maestro-v3.0.0.json           # MAESTRO dataset metadata (download separately, updated with tempo)
├── piano_rolls/                  # Directory to store processed .npy piano rolls and scaler metadata
│   └── scalers.npy               # Stores StandardScaler instances for inverse scaling
├── best_model.pth                # Saved weights of the best trained model
├── generated_music/              # (Optional) Directory for generated MIDI outputs
├── extract.py                    # Script to consolidate MIDI files
├── midi2pianoroll.py             # Converts MIDI to piano roll (.npy) and applies scaling
├── metadata.py                   # Extracts tempo from MIDI and updates JSON metadata
├── vectorization.py              # Handles text vectorization, feature scaling, and data pairing
├── model.py                      # Defines the PianoRollTransformer and PositionalEncoding classes
├── training_function.py          # Contains the `train_model` function for the Transformer
├── training.py                   # Main script to run model training
├── postprocessing.py             # Functions for inverse scaling and converting piano rolls to MIDI
└── README.md                     # This file


4. Data Preparation
The projec uses the MAESTRO dataset.

Dataset Acquisition
Download MAESTRO Dataset: Download maestro-v3.0.0.csv and maestro-v3.0.0.json from the MAESTRO dataset official page. Place them in your project root directory.

Download MIDI Files: Download the full MIDI dataset from MAESTRO. The MIDI files are typically organized in nested subfolders (e.g., maestro-v3.0.0/2004/MIDI-Unprocessed_01_R1_2004_mid--AUDIO_01_R1_2004_RNN_Unprocessed_SMF_01.midi).

MIDI File Consolidation
Run extract.py to move all .midi files from the nested MAESTRO dataset structure into a single dataset midi folder.

Bash

python extract.py
This will create the dataset midi directory and populate it.

MIDI to Piano Roll Conversion
Run midi2pianoroll.py to convert all MIDI files in dataset midi into NumPy piano roll arrays. These will be saved as .npy files in the piano_rolls directory. This script also applies StandardScaler to each piano roll and saves the individual scalers in piano_rolls/scalers.npy for inverse scaling later.

Bash

python midi2pianoroll.py
This will create the piano_rolls directory.

Metadata Augmentation (Tempo)
Run metadata.py to extract tempo information from each MIDI file in dataset midi and update the maestro-v3.0.0.json file.

Bash

python metadata.py
This script modifies maestro-v3.0.0.json in-place.

Text and Feature Vectorization
Run vectorization.py to:

Load and clean metadata from maestro-v3.0.0.csv.

Preprocess numerical features (tempo) using StandardScaler.

Preprocess categorical features (composer, title) using OneHotEncoder.

Generate text descriptions and create TF-IDF embeddings.

Pair the processed piano rolls, text embeddings, and structured features.

Split the data into training, validation, and test sets.

Save all processed data and crucial feature metadata (scalers, encoders, TF-IDF model) to maestro_text_to_pianoroll_processed.pt.

Bash

python vectorization.py
This step is crucial as it prepares the maestro_text_to_pianoroll_processed.pt file, which is the direct input for model training.


5. Model Training
After data preparation, you can train your Transformer model.

Review training.py: Open training.py and ensure that the text_vocab_size and structured_feature_dim hyperparameters are correctly set based on your vectorization.py output.

text_vocab_size: This depends on the max_features you set in TfidfVectorizer (or your actual text vocabulary size if using a different tokenization).

structured_feature_dim: This is the sum of your numerical features and the dimensionality after one-hot encoding your categorical features. You can determine this by inspecting the shape of structured_features after running vectorization.py.

num_patches: Corresponds to the maximum length of your piano rolls, which you set in midi2pianoroll.py (fs * duration) or determined from your dataset.

Run Training:

Bash

python training.py
This script will load the processed data, initialize the PianoRollTransformer, set up the optimizer and loss functions, and start the training loop defined in training_function.py. The best performing model weights (based on validation loss) will be saved as best_model.pth.


6. Music Generation (Inference)
To generate music from a text prompt:

Ensure best_model.pth exists: This file is created after successful training.

Run the generation script: Use the provided code logic (from the last response) to set up your inference pipeline. This will involve:

Loading the trained model and the feature_metadata (including tfidf, scaler, encoder) from maestro_text_to_pianoroll_processed.pt.

Processing your input text string using the same TF-IDF vectorizer and feature encoders/scalers used during training.

Calling the generate_piano_roll function with your model and prepared inputs.

Applying inverse_scale_piano_roll from postprocessing.py to the generated raw piano roll.

Using piano_roll_to_midi from postprocessing.py to convert the final piano roll into a MIDI file.

An example snippet for the generation part, assuming you've integrated it into a script like generate.py:

Python

# Example usage from a generation script (e.g., generate.py)
from your_model_definition import PianoRollTransformer
from postprocessing import inverse_scale_piano_roll, piano_roll_to_midi
import torch
import numpy as np
import re # for cleaning text
import torch.nn.functional as F # for padding

# Load processing metadata
processed_data = torch.load('maestro_text_to_pianoroll_processed.pt')
feature_metadata = processed_data['feature_metadata']
tfidf = feature_metadata['tfidf']
scaler = feature_metadata['scaler']
encoder = feature_metadata['encoder']
numerical_feature_names = feature_metadata['numerical_features']
categorical_column_names = feature_metadata['categorical_columns']

# Model hyperparameters (MUST match training)
# You might want to save these in feature_metadata for easier loading
text_vocab_size = tfidf.vocabulary_.__len__() # Example: if using raw TF-IDF for vocab
# structured_feature_dim calculation from vectorization.py:
# numerical_features.shape[1] + categorical_encoded.shape[1]
structured_feature_dim = len(numerical_feature_names) + len(categorical_column_names)

max_seq_len = 20 # Example, confirm with your vectorization.py config
num_patches = 400 # Example, confirm with your midi2pianoroll.py config (fs * duration)
num_pitches = 128
embedding_dim = 256
nhead = 8
num_encoder_layers = 4
num_decoder_layers = 4
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model
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
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Generation function (from previous responses)
def generate_piano_roll(model, src_text, structured_features, max_length, start_pitch, device):
    # ... (Same as the generate_piano_roll function provided in previous responses)
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


# Example Usage:
input_text = "mozart sonata A minor 100 tempo"
print(f"Generating music for: \"{input_text}\"")

# Preprocess Input Text
cleaned_text = re.sub(r'[^\\w\\s]', '', input_text.lower()).strip()
text_embedding_np = tfidf.transform([cleaned_text]).toarray()
text_tensor = torch.tensor(text_embedding_np, dtype=torch.float32).to(device)
# Pad if necessary for max_seq_len (TF-IDF output size might be fixed by max_features)
# If your model's positional_encoding expects specific padded length:
# padded_text_tensor = F.pad(text_tensor, (0, max_seq_len - text_tensor.shape[1])).to(device)
# Then use padded_text_tensor in model call


# Extract Structured Features (consistent with vectorization.py)
structured_features_input = np.zeros((1, structured_feature_dim))

# Handle tempo
tempo_match = re.search(r'(\\d+)\\s*tempo', input_text.lower())
tempo = int(tempo_match.group(1)) if tempo_match else 120
# Reshape for scaler and apply (tempo is numerical_features[0])
tempo_scaled = scaler.transform(np.array([[tempo]]))[0, 0]
structured_features_input[0, 0] = tempo_scaled # Assuming tempo is the 0th numerical feature

# Handle categorical features (e.g., key/composer from the text)
# This requires careful mapping to your one-hot encoded columns
# Example for 'minor' key: find the index for 'canonical_title_minor' or 'canonical_composer_minor'
key_terms = ["minor", "major"]
detected_key = None
for term in key_terms:
    if term in cleaned_text:
        detected_key = term
        break

if detected_key:
    # This part is highly dependent on how encoder was trained.
    # You'd need to create a dummy array for encoder.transform
    # or have a direct mapping from 'minor'/'major' to a one-hot index.
    # For simplicity, if 'minor' corresponds to a specific column:
    # Assuming canonical_title or canonical_composer contains 'minor'/'major'
    # e.g., if 'canonical_composer_mozart' and 'canonical_title_sonata a minor' are columns
    # You need to manually map these extracted terms to the one-hot indices.
    # A more robust solution might be to have pre-defined 'query' categorical features.

    # For example, if you know 'canonical_title_a minor' is a column:
    for i, col_name in enumerate(categorical_column_names):
        if f"canonical_title_a {detected_key}" in col_name: # Careful with exact string match!
            structured_features_input[0, len(numerical_feature_names) + i] = 1
            break
    # Similarly for composer "mozart"
    for i, col_name in enumerate(categorical_column_names):
        if "canonical_composer_mozart" == col_name:
            structured_features_input[0, len(numerical_feature_names) + i] = 1
            break


structured_tensor = torch.tensor(structured_features_input, dtype=torch.float32).to(device)


# Generate piano roll
max_gen_len = 400 # Or whatever `num_patches` is
start_pitch = 60.0 # A common starting pitch (e.g., C4)
generated_roll_raw = generate_piano_roll(model, text_tensor, structured_tensor, max_gen_len, start_pitch, device)

# Post-process and save MIDI
final_piano_roll = inverse_scale_piano_roll(generated_roll_raw, scaler=scaler) # Pass the loaded scaler
midi_output = piano_roll_to_midi(final_piano_roll, fs=100) # Ensure fs matches your midi2pianoroll.py
output_filename = "generated_mozart_sonata_A_minor_100_tempo.mid"
midi_output.write(output_filename)
print(f"Generated MIDI saved to {output_filename}")


8. Customization and Further Development
Hyperparameter Tuning: Experiment with embedding_dim, nhead, number of layers, dropout, learning rate, and batch size for optimal performance.

Loss Functions: Explore different loss functions or weighting strategies for pitch and velocity.

Decoding Strategy: Implement and test different decoding strategies like beam search or top-k/top-p sampling for generation, instead of simple greedy decoding.

Piano Roll Representation: Consider more complex piano roll representations (e.g., incorporating note duration explicitly, or using event-based MIDI representations).

Structured Features: Expand the types of structured features extracted from text (e.g., genre, mood, instrumentation).

Larger Vocabulary/Different Text Embeddings: If TF-IDF isn't sufficient, consider word embeddings (Word2Vec, GloVe) or contextual embeddings (BERT, etc.).

Dataset Expansion: Train on a larger or more diverse music dataset.

User Interface: Develop a simple web interface for text input and MIDI output.

