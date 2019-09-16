import reCVcable
import argparse
from os.path import isfile
from os import remove, listdir
from cv2 import cv2
from tensorflow.python.keras.models import load_model


 # Types of categories
classes = ['battery', 'disc', 'glass', 'metals', 'paper', 
        'plastic_jug_bottle', 'plastic_packaging', 'styrofoam']

ap = argparse.ArgumentParser()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")

imageArr = reCVcable.getDatabase()
imageArr = reCVcable.resizeImage(imageArr)
reCVcable.trainImages(imageArr, args["model"])
model = load_model(args["model"])
img = cv2.imread(args["image"])
pred, probability = reCVcable.predict_image(img, model)
print('%s %d%%' % (classes[pred], round(probability, 2) * 100))
_, ax = plt.subplots(1)
plt.imshow(reCVcable.cvtRGB(img))

ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()