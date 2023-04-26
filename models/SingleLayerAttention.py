import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLayerAttention(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(SingleLayerAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define a fully connected linear layer to project input to hidden size
        self.fc = nn.Linear(input_size, hidden_size)
        
        # Define another linear layer to calculate attention weights
        self.attn_weights = nn.Linear(hidden_size, 1)
    
    def forward(self, inputs):
        
        # Project inputs to hidden size using the fully connected layer
        x = self.fc(inputs)
        
        # Compute attention scores using the tanh function on x and the attention weights layer
        attn_scores = self.attn_weights(torch.tanh(x))
        
        # Apply softmax activation function to calculate attention weights
        attn_weights = F.softmax(attn_scores, dim=1) 
        
        # Compute attention output by taking a weighted sum of the input using the attention weights
        attn_output = torch.sum(attn_weights * x, dim=1)
        
        # Return attention output
        return attn_output


class AttentionClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionClassifier, self).__init__()
        
        # Define a single layer attention module
        self.attention = SingleLayerAttention(input_size, hidden_size)
        
        # Define a fully connected linear layer to project attention output to number of classes
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, inputs):
        
        # Compute attention output using the single layer attention module
        attn_output = self.attention(inputs)
        
        # Project attention output to number of classes using the fully connected layer
        logits = self.fc(attn_output)
        
        # Return logits
        return logits

class Trainer:
    
    def __init__(self, model, optimizer):
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
    
    # Train method to train the model
    def train(self, train_X, train_Y, num_epochs, batch_size):
        
        for epoch in range(num_epochs):
            
            train_loss = 0.0
            train_acc = 0.0
            
            for i in range(0, len(train_X), batch_size):
                
                # Get batch inputs and labels
                inputs, labels = train_X[i:i+batch_size], train_Y[i:i+batch_size]
                
                # Move inputs and labels to device
                inputs = torch.Tensor(inputs).to(self.device)
                labels = torch.Tensor(labels).to(self.device)
                
                # Compute logits using the model
                logits = self.model(inputs)
                
                # Compute cross-entropy loss between logits and labels
                loss = F.cross_entropy(logits, labels)
                
                # Reset optimizer gradients
                self.optimizer.zero_grad()
                
                # Compute gradients of loss with respect to model parameters
                loss.backward()
                
                # Update model parameters using optimizer
                self.optimizer.step()
                
                # Add batch loss and accuracy to train_loss and train_acc
                train_loss += loss.item() * inputs.shape[0]
                train_acc += (logits.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            
            # Compute epoch average train_loss and train_acc
            train_loss /= len(train_X)
            train_acc /= len(train_X)
            
            print(f'Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}')
