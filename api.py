from fastapi import FastAPI ,File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel('gemini-pro')

def get_details(class_name,ismedicinal):
    if ismedicinal== True:
        message = 'Give me five line information, highlighting its uses and significance, about this medicinal plant: ' + str(class_name)
        details = model.generate_content(message)
        return details.candidates[0].content.parts[0].text
    else:
        return None

model_ld = tf.keras.layers.TFSMLayer('../model/mpim_final', call_endpoint='serving_default')

app = FastAPI()

class_names = ['Aloevera',
 'Amla',
 'Amruthaballi',
 'Arali',
 'Astma_weed',
 'Badipala',
 'Balloon_Vine',
 'Bamboo',
 'Beans',
 'Betel',
 'Bhrami',
 'Bringaraja',
 'Caricature',
 'Castor',
 'Catharanthus',
 'Chakte',
 'Chilly',
 'Citron lime (herelikai)',
 'Coffee',
 'Common rue(naagdalli)',
 'Coriender',
 'Curry',
 'Doddpathre',
 'Drumstick',
 'Ekka',
 'Eucalyptus',
 'Ganigale',
 'Ganike',
 'Gasagase',
 'Ginger',
 'Globe Amarnath',
 'Guava',
 'Henna',
 'Hibiscus',
 'Honge',
 'Insulin',
 'Jackfruit',
 'Jasmine',
 'Kambajala',
 'Kasambruga',
 'Kohlrabi',
 'Lantana',
 'Lemon',
 'Lemongrass',
 'Malabar_Nut',
 'Malabar_Spinach',
 'Mango',
 'Marigold',
 'Mint',
 'Neem',
 'Nelavembu',
 'Nerale',
 'Nooni',
 'Onion',
 'Padri',
 'Palak(Spinach)',
 'Papaya',
 'Parijatha',
 'Pea',
 'Pepper',
 'Pomoegranate',
 'Pumpkin',
 'Raddish',
 'Rose',
 'Sampige',
 'Sapota',
 'Seethaashoka',
 'Seethapala',
 'Spinach1',
 'Tamarind',
 'Taro',
 'Tecoma',
 'Thumbe',
 'Tomato',
 'Tulsi',
 'Turmeric',
 'ashoka',
 'camphor',
 'kamakasturi',
 'kepala']

def image_ndarray(data):
    image = Image.open(BytesIO(data))
    image = image.resize((224,224))
    np_img = np.array(image)
    return np_img

@app.get("/ping")
async def ping():
    return "Hello I am Alive"

@app.post("/predict/")
async def predict(file: UploadFile):
    image = image_ndarray(await file.read())
    img_bth =  np.expand_dims(image,0)
    predictions = model_ld(img_bth)
    print(predictions["dense"])
    confidence = np.max(predictions["dense"])
    predict_class = class_names[np.argmax(predictions["dense"])]
    is_medicinal = True if confidence > 0.5 else False
    details = get_details(predict_class,is_medicinal)
    return {
            'class': predict_class,
            'confidence':float(confidence),
            'ismedicinal':is_medicinal,
            'details':details
            }

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)



