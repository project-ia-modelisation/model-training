import os
import torch
import trimesh
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import train_model
from script.evaluate import evaluate_model, load_preprocessed_model
from script.generate import generate_and_save_model
from models.model import Simple3DGenerator, Simple2DGenerator
from visualisation.lecture import read_image, evaluate_image, save_image

def train_pipeline(initial_model_files, num_iterations=10, num_generated_models=10, is_2d=False):
    for iteration in range(num_iterations):
        print(f"=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # Step 1: Train the model
        print("Étape 1 : Entraînement du modèle...")
        train_model(initial_model_files, is_2d=is_2d)
        print("Entraînement terminé.")
        
        # Step 2: Generate new models
        print("Étape 2 : Génération de nouveaux modèles...")
        model = Simple3DGenerator() if not is_2d else Simple2DGenerator()
        model.load_state_dict(torch.load("./data/model.pth" if not is_2d else "./data/model_2d.pth"))
        model.eval()
        generate_and_save_model(model, num_models=num_generated_models, is_2d=is_2d)
        print("Génération terminée.")
        
        # Step 3: Evaluate the generated models
        print("Étape 3 : Évaluation des modèles générés...")
        for i in range(num_generated_models):
            generated_model_path = f"./data/generated_model_{i}.obj" if not is_2d else f"./data/generated_model_{i}.png"
            generated_model = load_preprocessed_model(generated_model_path) if not is_2d else read_image(generated_model_path)
            
            # Utiliser tous les fichiers .obj ou .png comme modèles de vérité terrain
            ground_truth_files = [os.path.join("./data", f) for f in os.listdir("./data") if (f.endswith(".obj") if not is_2d else f.endswith(".png")) and not f.startswith("generated_model_")]
            for ground_truth_file in ground_truth_files:
                ground_truth_model = trimesh.load(ground_truth_file, force="mesh") if not is_2d else read_image(ground_truth_file)
                metrics = evaluate_model(generated_model, ground_truth_model) if not is_2d else evaluate_image(generated_model, ground_truth_model)
                print(f"Évaluation des métriques pour le modèle {i} avec {ground_truth_file}: {metrics}")
        
        # Update the list of model files for the next iteration
        initial_model_files = [os.path.join("./data", f) for f in os.listdir("./data") if (f.endswith(".obj") if not is_2d else f.endswith(".png"))]

if __name__ == "__main__":
    is_2d = False
    initial_model_files = [os.path.join("./data", f) for f in os.listdir("./data") if (f.endswith(".obj") if not is_2d else f.endswith(".png"))]
    train_pipeline(initial_model_files, num_iterations=10, num_generated_models=10, is_2d=is_2d)