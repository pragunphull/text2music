import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import os

# --- Step 1: Load and Clean Metadata ---

metadata = pd.read_csv('maestro-v3.0.0.csv')

# Keep relevant features (adjust based on your CSV)

metadata = metadata[['midi_filename', 'canonical_composer', 'canonical_title', 'tempo']]

numerical_features = ['tempo']
categorical_features = ['canonical_composer', 'canonical_title']

# Preprocess numerical features

scaler = StandardScaler()
metadata[numerical_features] = scaler.fit_transform(metadata[numerical_features])

# Preprocess categorical features

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_encoded = encoder.fit_transform(metadata[categorical_features])
categorical_cols = encoder.get_feature_names_out(categorical_features)

# Combine all features

structured_features = np.hstack([
    metadata[numerical_features].values,
    categorical_encoded
])


metadata['description'] = metadata['canonical_composer'] + " " + metadata['canonical_title']
metadata['description'] = metadata['description'].apply(
    lambda x: re.sub(r'[^\\w\\s]', '', x.lower()).strip()  # Clean text
)



piano_rolls = {}
for midi_file in metadata['midi_filename']:
    try:
        npy_file = midi_file.replace('.midi', '.npy')  # Adjust extension if needed
        piano_rolls[midi_file] = np.load(f'piano_rolls/{npy_file}')
    except FileNotFoundError:
        print(f"Warning: Piano roll not found for {midi_file}. Skipping.")
        continue

# Filter metadata to only include files with piano rolls

metadata = metadata[metadata['midi_filename'].isin(piano_rolls.keys())]

# --- Step 3: Train TF-IDF Model ---

tfidf = TfidfVectorizer(
    max_features=128,      # Reduce dimensionality
    stop_words='english',  # Remove common words
    ngram_range=(1, 2)     # Capture word pairs (e.g., "c major")
)
text_embeddings = tfidf.fit_transform(metadata['description']).toarray()  # Train TF-IDF


# --- Step 4: Pair Data ---

paired_data = []
for idx, row in metadata.iterrows():
    midi_file = row['midi_filename']
    if midi_file in piano_rolls: # added this check
        paired_data.append({
            'piano_roll': piano_rolls[midi_file],
            'text_embedding': text_embeddings[idx],            # From TF-IDF
            'structured_features': structured_features[idx], # Numerical + categorical
            'description': row['description']                 # Raw text (optional)
        })

paired_data = []
for idx, row in metadata.iterrows():
    midi_file = row['midi_filename']
    paired_data.append({
        'piano_roll': piano_rolls[midi_file],
        'text_embedding': text_embeddings[idx],          # From TF-IDF
        'structured_features': structured_features[idx], # Numerical + categorical
        'description': row['description']               # Raw text (optional)
    })

# --- Step 5: Train/Val/Test Split ---

train_data, temp_data = train_test_split(paired_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# --- Step 6: Save Processed Data & Metadata ---

torch.save({
    'train': train_data,
    'val': val_data,
    'test': test_data,
    'feature_metadata': {  # For inference
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'categorical_columns': categorical_cols,
        'tfidf_vocab': tfidf.get_feature_names_out(),
        'scaler': scaler,
        'encoder': encoder,
        'tfidf': tfidf  # Save the trained TF-IDF model
    }
}, 'maestro_text_to_pianoroll_processed.pt')
