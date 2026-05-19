from flask import Flask, render_template, redirect, request, url_for, send_file, session
from flask import jsonify, json
from werkzeug.utils import secure_filename

import mysql.connector

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.secret_key = 'abcdef'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    charset="utf8",
    database="deepfake"
)




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login',methods=['POST','GET'])
def login():
    
    
    msg=""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM user WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        
        if account:
            session['username'] = username
            session['user_type'] = 'user'
            msg="success"
            return redirect(url_for('upload'))
        else:
            msg="fail"
        

    return render_template('login.html',msg=msg)


@app.route('/register',methods=['POST','GET'])
def register():
    msg=""
    st=""
    name=""
    email=""
    mess=""
    reg_no=""
    password=""
    if request.method=='POST':

        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        username=request.form['username']
        password=request.form['password']

        
        
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM user where username=%s",(username, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM user")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO user(id, name, mobile, email, username, password) VALUES (%s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, username, password)
            mycursor.execute(sql, val)
            mydb.commit()

            msg="success"
            return redirect(url_for('login'))
            st="1"
            mess = f"Reminder: Hi {name}, your username is {reg_no} and password is {password}!"
            mycursor.close()
        else:
            msg="fail"
            
    return render_template('register.html', msg=msg)




# Creating Model Architecture

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      faces = face_recognition.face_locations(frame)
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image

#Capsule Networks -accurate detection of manipulated visual content
class CapsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=9)
        self.primary_caps = nn.Conv2d(256, 32 * 8, kernel_size=9)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.primary_caps(x)
        return x.view(x.size(0), -1)

class SiameseNetwork():
    def __init__(self):
        super().__init__()
        self.capsnet = CapsNet()

    def forward(self, img1, img2):
        f1 = self.capsnet(img1)
        f2 = self.capsnet(img2)
        return f1, f2

def contrastive_loss(f1, f2, label, margin=1.0):
    dist = F.pairwise_distance(f1, f2)
    loss = torch.mean((1-label)*dist**2 +
                      label*torch.clamp(margin-dist, min=0)**2)
    return loss

def detectFakeImage(imagePath):
    # Load the InceptionNet model
    model = load_model('model/inceptionNet_model.h5')
    
    # Model parameters
    target_size = (299, 299)  # InceptionNet default input size
    
    # Read and preprocess image
    img = cv2.imread(imagePath)
    if img is None:
        return ["ERROR: Could not read image", 0.0]
        
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find faces in the image
    faces = face_recognition.face_locations(img)
    
    if not faces:
        return ["NO FACE DETECTED", 0.0]
        
    # Use the first face found
    top, right, bottom, left = faces[0]
    
    # Add padding to the face region
    face_padding = 20
    top = max(0, top - face_padding)
    right = min(img.shape[1], right + face_padding)
    bottom = min(img.shape[0], bottom + face_padding)
    left = max(0, left - face_padding)
    
    # Extract and resize face
    face = img[top:bottom, left:right]
    face = cv2.resize(face, target_size)
    
    # Preprocess for InceptionNet
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    
    # Predict
    probs = model.predict(face)[0]
    pred = np.argmax(probs)
    confidence = probs[pred] * 100
    result = "REAL" if pred == 1 else "FAKE"
    
    print(f"Image Prediction: {result}, Confidence: {confidence:.2f}%")
    return [result, confidence]

# [Rest of the Flask app code remains the same]



@app.route('/upload', methods=['POST', 'GET'])
def upload():


  
  return render_template('upload.html')


import os

from werkzeug.utils import secure_filename


import platform

import os
import platform
import mimetypes
import requests
from werkzeug.utils import secure_filename

def get_geo_info(ip):
    try:
        res = requests.get(f"https://ipapi.co/{ip}/json/").json()
        lat = res.get("latitude")
        lon = res.get("longitude")
        return lat, lon
    except:
        return None, None

@app.route('/detect', methods=['POST', 'GET'])
def DetectPage():
    if 'file' not in request.files:
        return render_template('upload.html', data="No file selected!")

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('upload.html', data="No file selected!")

    # Ensure upload folder exists
    upload_folder = os.path.join("static", "Uploaded_Files")
    os.makedirs(upload_folder, exist_ok=True)

    # Save the uploaded file
    file_name = secure_filename(uploaded_file.filename)
    file_path = os.path.join(upload_folder, file_name)
    uploaded_file.save(file_path)

    # Detect type and run model
    
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        prediction = detectFakeImage(file_path)
        file_type = 'image'
    else:
        return render_template('upload.html', data="Unsupported file format!")

    result, confidence = prediction[0], prediction[1]

    # Get system and user info
    ip_address = request.remote_addr
    browser_info = request.headers.get('User-Agent')
    os_info = platform.system() + " " + platform.release()

    # Extra metadata
    file_size = os.path.getsize(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    lat, lon = get_geo_info(ip_address)

    # Get user ID
    username = session.get('username')
    cursor = mydb.cursor()
    cursor.execute("SELECT id FROM user WHERE username = %s", (username,))
    user = cursor.fetchone()

    if user:
        user_id = user[0]

        

    return render_template('result.html', uploaded_file=file_path, result=result, confidence=confidence)

@app.route('/view_uploads')
def view_uploads():

    username=session.get('username')
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM media_uploads where username=%s", (username, ))
    uploads = cursor.fetchall()
    return render_template('view_uploads.html', uploads=uploads)



import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define MesoNet architecture
class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def detectFakeImage(imagePath):
    # Load model
    model = Meso4()
    model.load_state_dict(torch.load('model/mesonet_model.pth', map_location=torch.device('cpu')), strict=False)

    model.eval()

    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    # Load and preprocess image
    image = Image.open(imagePath).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 112, 112)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred]) * 100
        result = "REAL" if pred == 1 else "FAKE"

    print(f"Image Prediction: {result}, Confidence: {confidence:.2f}%")
    return [result, confidence]



@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True
    app.run()
