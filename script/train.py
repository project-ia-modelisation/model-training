import os
import numpy as np
import torch
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader, Dataset
from models.model import Simple3DGenerator, Simple2DGenerator
from data_processing.scripts.preprocess import load_and_preprocess_model
from data_processing.scripts.prompt_handler import PromptHandler 

class Model3DDataset(Dataset):
    def __init__(self, model_files, num_vertices=1000):
        self.models = []
        for file in model_files:
            try:
                model = load_and_preprocess_model(file, num_vertices)
                vertices = np.array(model.vertices, dtype=np.float32)
                self.models.append(vertices)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle {file}: {e}")

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        return torch.tensor(self.models[idx])

class Model2DDataset(Dataset):
    def __init__(self, image_files):
        self.images = []
        for file in image_files:
            try:
                image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Erreur lors de la lecture de l'image {file}")
                image = cv2.resize(image, (64, 64))  # Assuming 64x64 image size
                self.images.append(image)
            except Exception as e:
                print(f"Erreur lors du chargement de l'image {file}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension

def train_model(model_files, num_epochs=10, prompt=None, batch_size=8, num_vertices=1000, learning_rate=0.001, is_2d=False):
    prompt_handler = PromptHandler()
    if prompt:
        params = prompt_handler.appliquer_prompt(prompt)
        num_vertices = params["nb_sommets"]
    else:
        num_vertices = 1000 
        
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Création du dataset et dataloader
    try:
        dataset = Model3DDataset(model_files, num_vertices) if not is_2d else Model2DDataset(model_files)
        if len(dataset) == 0:
            raise ValueError("Aucun modèle n'a pu être chargé")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        raise Exception(f"Erreur lors de la création du dataset : {e}")

    # Initialisation du modèle et optimiseur
    model = Simple3DGenerator(num_vertices=num_vertices).to(device) if not is_2d else Simple2DGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    print(f"Début de l'entraînement avec {num_vertices} sommets..." if not is_2d else "Début de l'entraînement pour les plans 2D...")

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            inputs = torch.randn(batch.size(0), 100).to(device)  # Exemple d'entrée aléatoire
            targets = batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).view(batch.size(0), num_vertices, 3) if not is_2d else model(inputs).view(batch.size(0), -1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Sauvegarder le modèle entraîné
    torch.save(model.state_dict(), "./data/model.pth" if not is_2d else "./data/model_2d.pth")
    print("Modèle entraîné sauvegardé avec succès." if not is_2d else "Modèle 2D entraîné sauvegardé avec succès.")
    
model = torch.nn.Linear(10, 1)
torch.save(model.state_dict(), "model.pth")