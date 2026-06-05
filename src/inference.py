import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

def predict_image(model_path, class_names, image_path):

    model = load_model(model_path)

    img = load_img(image_path, target_size=(224,224))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)

    return class_names[idx], float(np.max(pred))
