import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection, AutoTokenizer


class ClipTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32") # loading the pre trained encoder
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # We don't want to train the pretrained model. This freezes the model
        for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval() # set to eval mode

        self.reduction_layer = nn.Linear(self.model.config.projection_dim, text_embedding_size) # reduces to embedding size we want,
        # clips embedding is different to our size, this is the only layer the model is trained and weights change

    def forward(self, text):
        embedding = self.encode(text) # embeds the text
        return self.reduction_layer(embedding) # reduces to our embed size

    def encode(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt", truncation=True,
                max_length=77)  # tokenizes the text inputs
        # We need to do this so that all the inputs are in the same device
        inputs = {key: value.to(device) for key, value in inputs.items()} # Move inputs to the correct device

        with torch.no_grad(): # runs model without tracking gradients
            # The "**" unpacks the dictionary.
            outputs = self.model(**inputs)

        return outputs.text_embeds
