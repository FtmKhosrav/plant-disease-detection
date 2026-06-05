import os
import shutil
import random

test_ratio = 0.15
val_ratio = 0.15


def remove_folders(base_path):
    for folder in ['Train', 'Validation', 'Test']:
        path = os.path.join(base_path, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted {folder}")


def split_data_for_plant(plant_path, output_base_path):

    plant_name = os.path.basename(plant_path)

    train_path = os.path.join(output_base_path, plant_name, "Train")
    val_path = os.path.join(output_base_path, plant_name, "Validation")
    test_path = os.path.join(output_base_path, plant_name, "Test")

    for p in [train_path, val_path, test_path]:
        os.makedirs(p, exist_ok=True)

    for disease in sorted(os.listdir(plant_path)):

        disease_path = os.path.join(plant_path, disease)

        if not os.path.isdir(disease_path):
            continue

        files = [f for f in os.listdir(disease_path)
                 if f.lower().endswith(('.jpg','.png','.jpeg'))]

        random.shuffle(files)

        test_count = int(len(files) * test_ratio)
        val_count = int(len(files) * val_ratio)

        test_files = files[:test_count]
        val_files = files[test_count:test_count + val_count]
        train_files = files[test_count + val_count:]

        for f in train_files:
            shutil.copy(os.path.join(disease_path, f),
                        os.path.join(train_path, disease, f))

        for f in val_files:
            shutil.copy(os.path.join(disease_path, f),
                        os.path.join(val_path, disease, f))

        for f in test_files:
            shutil.copy(os.path.join(disease_path, f),
                        os.path.join(test_path, disease, f))
