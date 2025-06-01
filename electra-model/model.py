import torch
import random


class ElectraModel(torch.nn.Module): 
    def __init__(self, generator, discriminator): 
        self.generator = generator
        self.discriminator = discriminator


    def forward(self, input, vocab): 
        mask_tensor = [random.random() < 0.15 for _ in range(len(input))]
        
