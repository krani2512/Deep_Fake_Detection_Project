import cv2
import numpy as np
import os

os.makedirs("dataset/real", exist_ok=True)
os.makedirs("dataset/fake", exist_ok=True)

def gen(img_path, text, color):
    img = np.zeros((224,224,3), dtype=np.uint8)
    cv2.circle(img,(112,112),80,color,-1)
    cv2.putText(img,text,(70,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,
                (255,255,255),2)
    cv2.imwrite(img_path,img)

for i in range(1,51):
    gen(f"dataset/real/real_{i:03}.jpg","REAL",(0,255,0))
    gen(f"dataset/fake/fake_{i:03}.jpg","FAKE",(0,0,255))

print("Dataset Sample100 Created")
