import tensorflow as tf
import argparse
import PIL
import numpy as np
import json
import tensorflow_hub as hub



parser = argparse.ArgumentParser(description = 'Predict flower species')

parser.add_argument('path', action='store', help='Path to image')

parser.add_argument('--top_k', action='store', help='Desired number of most likely labels to be displayed', default = 1)

parser.add_argument('--category_names', action='store', help='Path to json file that maps labels to species names', default = None)

parser.add_argument('--mlmodel', action='store', help='Path to the desired machine learning model', default='Attempt_1.h5')

args = parser.parse_args()

path = str(args.path)
topk = int(args.top_k)
labelmap = str(args.category_names)
modelpath = str(args.mlmodel)

model = tf.keras.models.load_model(modelpath, custom_objects={'KerasLayer':hub.KerasLayer})

im = PIL.Image.open(path)
im = tf.convert_to_tensor(im, dtype=tf.float32)
im = tf.image.resize(im, (224, 224))
im /= 255
im = im.numpy()
expanded = tf.convert_to_tensor(np.expand_dims(im, axis=0))
probs, classes = tf.nn.top_k(model.predict(expanded)[0], k=topk)
probs = probs.numpy()
classes = classes.numpy()

if labelmap == 'None':
    for i in range(0,topk):
        print(str(classes[i]+1) + " " + str(probs[i]))
else:
    with open(labelmap, 'r') as f:
        class_names = json.load(f)
    
    named_classes = []

    for i in classes:
        named_classes.append(class_names[str(i+1)])
        
    for i in range(0,topk):
        print(str(named_classes[i]) + " " + str(probs[i]))