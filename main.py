import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------
# MODEL CONFIGURATION
# -------------------------------
models = {
    "Corn": {
        "path": "models/corn_model.h5",
        "classes": ['common_rust', 'grey_leaf_spot', 'healthy', 'northern_leaf_blight'],
        "solutions": {
            "healthy": "Plant is healthy.",
            "grey_leaf_spot": "Use fungicides like Mancozeb and reduce humidity.",
            "common_rust": "Apply Propiconazole fungicide and reduce leaf moisture.",
            "northern_leaf_blight": "Remove infected leaves and use systemic fungicides."
        }
    },

    "Tomato": {
        "path": "models/tomato_model.h5",
        "classes": [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites', 'Tomato_Target_Spot',
            'Tomato_YellowLeaf_Curl_Virus', 'Tomato_healthy'
        ],
        "solutions": {
            "Tomato_healthy": "Plant is healthy.",
            "Tomato_Bacterial_spot": "Use copper-based fungicides.",
            "Tomato_Early_blight": "Improve airflow and apply Mancozeb.",
            "Tomato_Late_blight": "Remove infected leaves and use systemic fungicides."
        }
    },

    "Rice": {
        "path": "models/rice_model.h5",
        "classes": [
            'Brown_Spot', 'Leaf_Blast', 'Neck_Blast', 'Healthy'
        ],
        "solutions": {
            "Healthy": "Plant is healthy.",
            "Brown_Spot": "Improve spacing and use fungicides.",
            "Leaf_Blast": "Avoid over-irrigation and apply systemic fungicides.",
            "Neck_Blast": "Use resistant seeds and proper field management."
        }
    },

    "Wheat": {
        "path": "models/wheat_model.h5",
        "classes": [
            'Brown_Rust', 'Yellow_Rust', 'Healthy'
        ],
        "solutions": {
            "Healthy": "Plant is healthy.",
            "Brown_Rust": "Use crop rotation and fungicides.",
            "Yellow_Rust": "Apply fungicide and improve field ventilation."
        }
    }
}

# -------------------------------
# IMAGE PREDICTION FUNCTION
# -------------------------------
def predict_image(model, class_names, image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return class_names[class_index], float(np.max(prediction))

# -------------------------------
# MAIN PROGRAM
# -------------------------------
print("Available plants: Corn, Tomato, Rice, Wheat")
plant = input("Select plant type: ")

if plant not in models:
    print("Invalid selection!")
    exit()

# Load model
model_info = models[plant]
model = load_model(model_info["path"])

print(f"{plant} model loaded successfully!")

# Image input
image_path = input("Enter image path: ")

# Prediction
label, confidence = predict_image(
    model,
    model_info["classes"],
    image_path
)

solution = model_info["solutions"].get(label, "No solution available")

# Output
print("\n========================")
print(f"Predicted Disease: {label}")
print(f"Confidence: {confidence:.2f}")
print(f"Solution: {solution}")
print("========================\n")

# Show image
img = load_img(image_path)
plt.imshow(img)
plt.axis("off")
plt.title(f"{label} ({confidence:.2f})")
plt.show()
