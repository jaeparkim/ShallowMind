# Import required packages
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Define a class for sentence embedding
class SentenceEmbedder:
    def __init__(self, model_name='paraphrase-distilroberta-base-v2'):
        # Load the pre-trained sentence embedding model
        self.embed_model = SentenceTransformer(model_name)
        # Set the device to use for the embedding model
        self.device = torch.device('cuda:0')
        # Move the model to the selected device
        self.embed_model.to(self.device)
        # Get the maximum sequence length of the model
        self.max_seq_length = self.embed_model.get_max_seq_length()
    
    def create_sentence_embedding(self, long_sentence):
        # Split the long sentence into segments of maximum sequence length
        segments = []
        for i in range(0, len(long_sentence), self.max_seq_length):
            segment = long_sentence[i:i+self.max_seq_length]
            segments.append(segment)

        # If the long sentence is empty, use a default segment "the end"
        if(len(segments) == 0):
            segments = ["the end"]

        # Encode each segment using the embedding model
        segment_embeddings = []
        for segment in segments:
            if(len(segment)>0):
                # Generate an embedding for the segment using the loaded model
                segment_embedding = self.embed_model.encode(segment)
                segment_embeddings.append(segment_embedding)

        # Compute the mean of all segment embeddings to get the final sentence embedding
        long_sentence_embedding = np.mean(segment_embeddings, axis=0)

        return long_sentence_embedding
    
    def embed_passage(self,datum):
        # Split the passage into segments separated by the _SEP_ token and generate embeddings for each segment
        return np.concatenate(np.vectorize(self.create_sentence_embedding, otypes=[np.ndarray])(datum.split("_SEP_")[0:10]))

    def generate_embeddings(self, train_X):
        # Generate sentence embeddings for all passages in train_X and stack them into a numpy array
        data_train_embedded = []
        for text in tqdm(train_X):
          vec = np.array(self.embed_passage(text))
          data_train_embedded.append(vec)
        data_train_embedded = np.stack(np.array(data_train_embedded))
        # Reshape the input data arrays into a 3D tensor with dimensions (batch_size, sequence_length, embedding_size)
        return data_train_embedded.reshape((data_train_embedded.shape[0], 10, 768))

    # Define a function called cosine_similarity that takes in two vectors a and b
    def cosine_similarity(self,a, b):
        # Calculate the dot product of vectors a and b
        dot_product = np.dot(a, b)
        # Calculate the norm of vector a
        norm_a = np.linalg.norm(a)
        # Calculate the norm of vector b
        norm_b = np.linalg.norm(b)
        # Calculate the cosine similarity of vectors a and b and return it
        return dot_product / (norm_a * norm_b)

    def generate_running_embedding(self,train_X):
        # Create an empty list called cosine_values to store the cosine similarity values
        cosine_values = []

        # Iterate over each text in train_X and track the progress using tqdm
        for text in tqdm(train_X):
          # Create an empty string called curr and an empty list called embeddings
          curr = ""
          embeddings = []
          # Iterate over each sentence in the text separated by "_SEP_" and concatenate it to curr
          # Then create an embedding for the concatenated sentence and append it to embeddings
          for stc in text.split("_SEP_"):
            curr += stc + " "
            embeddings.append(self.create_sentence_embedding(curr.strip()))
          # Calculate the cosine similarity between each pair of consecutive embeddings
          # and append the values to cosine_sim
          cosine_sim = []
          for idx in range(1,len(embeddings)):
            cosine_sim.append(self.cosine_similarity(embeddings[idx-1],embeddings[idx]))

          # Apply Gaussian filter
          sigma = 2
          cosine_sim = gaussian_filter(cosine_sim, sigma)
          # Append the list of cosine similarity values to cosine_values
          cosine_values.append(cosine_sim)

        # Return the list of cosine similarity values
        return cosine_values
    
    def eliminate_zero_boundry(self,x,y):

        filtered_x, filtered_y = [], []
        
        # iterate through each index of y
        for idx in range(len(y)):
            
            # check if the value of y at that index is greater than 0
            if y[idx] > 0:
                
                # if it is, append the first 9 elements of x at that index to filtered_x 
                filtered_x.append(x[idx][:9])
                
                # subtract 1 from the value of y at that index, and append it to filtered_y
                filtered_y.append(y[idx]-1)

        # return filtered_x and filtered_y as numpy arrays
        return np.array(filtered_x), np.array(filtered_y)


    def onehot_output(self,train_Y, val_Y, test_Y):
        # Convert the integer labels to one-hot encoded tensors using PyTorch's one_hot() function
        # The labels are converted to float tensors so that they can be used with PyTorch's loss functions
        train_Y = torch.nn.functional.one_hot(torch.from_numpy(train_Y), num_classes=10).float()
        val_Y = torch.nn.functional.one_hot(torch.from_numpy(val_Y), num_classes=10).float()
        test_Y = torch.nn.functional.one_hot(torch.from_numpy(test_Y), num_classes=10).float()

        return train_Y,val_Y,test_Y
