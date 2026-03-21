import torch
import torch.nn as nn
import torchvision.models as models

class ArtCNNRNN(nn.Module):
    def __init__(self, num_artists, num_styles, num_genres):
        super(ArtCNNRNN, self).__init__()
        
        # 1. Convolutional Encoder (ResNet18)
        # We use a lightweight CNN for rapid local training
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Strip the final pooling and classification layers
        # This leaves us with a feature map of shape (Batch, 512, 7, 7)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. Recurrent Spatial Analyzer (Bi-directional LSTM)
        # Input: 512 channels. We flatten the 7x7 grid into a sequence of 49 patches.
        self.rnn = nn.LSTM(
            input_size=512, 
            hidden_size=256, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True # Reads the canvas forward and backward
        )
        
        # --- NEW: Add Dropout ---
        self.dropout = nn.Dropout(p=0.4)
        
        # 3. Multi-Task Classification Heads
        # Bi-LSTM concatenates forward and backward states (256 * 2 = 512)
        self.artist_head = nn.Linear(512, num_artists)
        self.style_head = nn.Linear(512, num_styles)
        self.genre_head = nn.Linear(512, num_genres)

    def forward(self, x):
        # x shape: (Batch, 3, 224, 224)
        
        # Extract visual features
        features = self.cnn(x) # Output: (Batch, 512, 7, 7)
        
        # Flatten spatial dimensions to create a "sequence" of patches
        batch_size, channels, h, w = features.size()
        features = features.view(batch_size, channels, h * w) # (Batch, 512, 49)
        features = features.permute(0, 2, 1) # (Batch, Sequence=49, Features=512)
        
        # Pass the sequence of patches through the RNN
        rnn_out, (h_n, c_n) = self.rnn(features)
        
        # Extract the final hidden state from both directions
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        final_state = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # (Batch, 512)
        
        # --- NEW: Apply Dropout ---
        final_state = self.dropout(final_state)
        
        # Pass the final comprehensive thought into the multi-task heads
        artist_pred = self.artist_head(final_state)
        style_pred = self.style_head(final_state)
        genre_pred = self.genre_head(final_state)
        
        # We return the final_state as well because we need it for the t-SNE outlier plot!
        return artist_pred, style_pred, genre_pred, final_state