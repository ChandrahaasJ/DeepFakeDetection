from torchvision import transforms
from PIL import Image 

import pickle
import json
import requests
image_path = r"C:\Users\Chandrahaas\OneDrive\Pictures\Screenshots\Screenshot 2024-12-11 220452.png"
image = Image.open(image_path).convert("RGB")#replace image_path with the path of image in your PC
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
image = transform(image)

tensor=image.unsqueeze(0) 
pickle_bytesTF = pickle.dumps(tensor)
tensor_stringTF = pickle_bytesTF.decode('latin1')
dict={'tensor':tensor_stringTF}
x=json.dumps(dict)
res=requests.post("http://127.0.0.1:8000/predict",x)
print(res.json())