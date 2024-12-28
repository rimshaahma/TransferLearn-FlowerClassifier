import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('resnet50_model.h5')  # You can use any model here

# Define the class labels
class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def classify_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Resize the image to match input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = class_labels[predicted_class[0]]
    
    return predicted_label

# Set up the GUI window
window = tk.Tk()
window.title("Flower Image Classifier")

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        label.config(text="Classifying image...")
        result = classify_image(file_path)
        label.config(text=f"Predicted: {result}")
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# Create and place GUI elements
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

panel = tk.Label(window)
panel.pack()

label = tk.Label(window, text="Prediction will appear here")
label.pack()

window.mainloop()
