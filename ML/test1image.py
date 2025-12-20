import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K 
import  cv2

from PIL import Image, ImageChops, ImageEnhance
def predict():
 image_size = (128, 128)

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
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
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0
# Load the trained model


model = load_model('ML/model_multiclass.h5')  # Replace with the path to your trained model

class_names = ['anthracnose', 'quick_wilt', 'white_spots']
image_path = 'test.jpg'
#image_path = 'ML/test/test/test.jpg'
image = prepare_image(image_path)
#real_image_path = 'dataset/CASIA2/Au/Au_ani_00001.jpg'
#fake_image_path = 'dataset/CASIA2/Tp/Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg'
#image = prepare_image(fake_image_path)
image = image.reshape(-1, 128, 128, 3)
y_pred = model.predict(image)
y_pred_class = np.argmax(y_pred, axis = 1)
if y_pred_class[0]==0:
    out="a"
elif y_pred_class[0]==1:
    out="q"
else:
    out="w"
    #print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
    #return(out)
    print(out)

