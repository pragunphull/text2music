import torch

def generate_piano_roll(model, src_text, structured_features, max_length, device):
    """
    Generates a piano roll given a trained model, source text, and structured features.

    Args:
        model (PianoRollTransformer): The trained piano roll transformer model.
        src_text (torch.Tensor): The encoded source text (e.g., from TF-IDF). Shape: (batch_size, seq_len).
        structured_features (torch.Tensor): The structured features (e.g., composer, tempo). Shape: (batch_size, structured_feature_dim).
        max_length (int): The maximum length of the generated piano roll.
        device (torch.device): The device to use (CPU or CUDA).

    Returns:
        torch.Tensor: The generated piano roll. Shape: (batch_size, max_length, 2). 
                      The last dimension contains [pitch, velocity].
    """

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        batch_size = src_text.size(0)
        
        # Initialize the target piano roll with the start token(s)
        # Assuming '0' represents the start/no note token for pitch
        tgt_pianoroll = torch.zeros((batch_size, 1, 2), dtype=torch.float, device=device)  
        tgt_pianoroll[:, 0, 0] = 0  # Start pitch token
        tgt_pianoroll[:, 0, 1] = 0  # Start velocity (you might want to adjust this)

        for t in range(1, max_length):
            # Predict the next pitch and velocity
            pitch_predictions, velocity_predictions = model(src_text, structured_features, tgt_pianoroll)
            
            # Get the last time step's predictions
            last_pitch_logits = pitch_predictions[:, -1, :]  # (batch_size, num_pitches + 1)
            last_velocity = velocity_predictions[:, -1, :]   # (batch_size, 1)

            # Sample the pitch from the distribution
            pitch_probs = torch.softmax(last_pitch_logits, dim=-1)
            next_pitch = torch.multinomial(pitch_probs, num_samples=1).float()  # (batch_size, 1)

            # Or, you can deterministically pick the most likely pitch:
            # next_pitch = torch.argmax(last_pitch_logits, dim=-1).float().unsqueeze(-1)

            # Clip velocity to [0, 1] and scale it to your desired range if needed
            next_velocity = torch.clamp(last_velocity, 0, 1) 

            # Append the predictions to the target piano roll
            next_step = torch.cat((next_pitch, next_velocity), dim=-1).unsqueeze(1) # (batch_size, 1, 2)
            tgt_pianoroll = torch.cat((tgt_pianoroll, next_step), dim=1)

    return tgt_pianoroll

# --- Usage Example ---

# Assuming you have a trained 'model', and you have prepared your input:
# src_text (torch.Tensor), structured_features (torch.Tensor)

# Example: dummy data (replace with your actual data)
batch_size = 2
seq_len = 10
structured_feature_dim = 128  # Example value, get from your data
max_length = 400 # Length of the generated piano roll

src_text = torch.randint(0, 100, (batch_size, seq_len)).to(device) 
structured_features = torch.randn(batch_size, structured_feature_dim).to(device)


generated_pianoroll = generate_piano_roll(model, src_text, structured_features, max_length, device)

