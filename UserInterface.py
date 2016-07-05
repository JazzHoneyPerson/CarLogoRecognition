# Main.py
# TianYang Jin, Yu Fu
#
# An user interface that allows user
# to select image and gives prediction of car logo.

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import NN_predict

class App():
    def __init__(self, main):
        self.label = ttk.Label(main, text="Your car logo is: ")
        self.label.grid(column=3, row=1)

        self.canvas = Canvas(main, width=600, height=400)
        self.canvas.grid(row=5, column=3)

        white_img = np.ones((600, 400), dtype=np.uint8) * 255
        self.photo = ImageTk.PhotoImage(Image.fromarray(white_img))
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
        self.button = Button(main, text="Choose Picture", command=self.open_image).grid(column=3, row=3)

    def open_image(self):
        img_path = filedialog.askopenfilename(filetypes=
                                              [("Image files", ("*.jpg", "*.jpeg", "*.bmp", "*.png"))])
        image = Image.open(img_path).resize((600, 400))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)

        self.label.config(text=("Your car logo is: " + self.predict(img_path)))
    
    '''
    Use different theta files to enable bootstrap aggregation.
    '''
    def predict(self, image_path):
        l = []
        l.append(NN_predict.predictImage(image_path, 'thetas1'))
        l.append(NN_predict.predictImage(image_path, 'thetas2'))
        l.append(NN_predict.predictImage(image_path, 'thetas3'))
        return max(set(l), key=l.count)

root = Tk()
App(root)
root.mainloop()
