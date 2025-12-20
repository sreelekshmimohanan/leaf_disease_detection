'''import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

# Define global image size
image_size = (128, 128)

def predict():
    def convert_to_ela_image(path, quality):
        temp_filename = 'temp_file_name.jpg'
        ela_filename = 'temp_ela.png'

        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality=quality)
        temp_image = Image.open(temp_filename)

        ela_image = ImageChops.difference(image, temp_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return ela_image

    def prepare_image(image_path):
        global image_size  # Use the global variable image_size
        return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

    # Load the trained model
    model = load_model('ML/model_multiclass.h5')
    class_names = ['anthracnose', 'quick_wilt', 'white_spots']
    
    image_path = 'ML/test/test/white_spots02215.jpg'  # Define the image path here
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    pred = model.predict(image)

    #print(pred)

    predicted_class_indices = np.argmax(pred, axis=1)
    print(predicted_class_indices)
    label = ['anthracnose', 'quick_wilt', 'white_spots']
    #print(label[predicted_class_indices[0]])

# Call the predict function
predict()'''
import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

# Define global image size
image_size = (128, 128)

def predict():
    def convert_to_ela_image(path, quality):
        temp_filename = 'temp_file_name.jpg'
        ela_filename = 'temp_ela.png'

        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality=quality)
        temp_image = Image.open(temp_filename)

        ela_image = ImageChops.difference(image, temp_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return ela_image

    def prepare_image(image_path):
        global image_size  # Use the global variable image_size
        return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

    # Load the trained model
    model = load_model('ML/model_multiclass.h5')
    class_names = ['anthracnose', 'quick_wilt', 'white_spots']
    
    image_path = 'ML/test/test/white_spots02215.jpg'  # Define the image path here
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    pred = model.predict(image)

   # print(pred)

    predicted_class_indices = np.argmax(pred, axis=1)
    #print(predicted_class_indices)
    label = ['anthracnose', 'quick_wilt', 'white_spots']
    print(label[predicted_class_indices[0]])

# Call the predict function
#predict()
