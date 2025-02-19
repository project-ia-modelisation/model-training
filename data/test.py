import trimesh

file_path = "./ground_truth_model.obj"

try:
    model = trimesh.load(file_path, force="mesh")
    print(f"✅ Modèle chargé depuis {file_path}")
    print(f"🔍 Nombre de sommets : {len(model.vertices)}")
    print(f"🔍 Nombre de faces : {len(model.faces)}")

    # Vérifier si des faces contiennent des indices invalides
    max_index = len(model.vertices)
    invalid_faces = []
    
    for i, face in enumerate(model.faces):
        if any(v >= max_index or v < 0 for v in face):
            invalid_faces.append((i, face))

    if invalid_faces:
        print(f"❌ {len(invalid_faces)} faces contiennent des indices invalides !")
        for index, face in invalid_faces[:10]:  # Afficher seulement 10 erreurs
            print(f"⚠️ Face {index} : {face}")
    else:
        print("✅ Tous les indices de faces sont valides.")
        
except Exception as e:
    print(f"❌ Erreur lors du chargement du fichier OBJ : {e}")
