import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

window = tk.Tk()
window.title('tool')
align_mode = 'nswe'
pad = 5
file_path = 'C:/Users/88691/數位影像處理/HW/HW3/monet2.jpg'
# window.geometry('600x600')


def define_layout(obj, cols=1, rows=1):

    def method(trg, col, row):

        for c in range(cols):
            trg.columnconfigure(c, weight=1)
        for r in range(rows):
            trg.rowconfigure(r, weight=1)

    if type(obj) == list:
        [method(trg, cols, rows) for trg in obj]
    else:
        trg = obj
        method(trg, cols, rows)


def callback(e):
    x = e.x
    y = e.y
    print("Pointer is currently at %d, %d" % (x, y))


def openfile():
    path = filedialog.askopenfilename()
    return path


div_size = 200
img_size = div_size * 2
div1 = tk.Frame(window, width=img_size, height=img_size, bg='blue')
div2 = tk.Frame(window, width=div_size, height=div_size, bg='orange')
div3 = tk.Frame(window, width=div_size, height=div_size, bg='green')

window.update()
win_size = min(window.winfo_width(), window.winfo_height())
print(win_size)

div1.grid(column=0, row=0, padx=pad, pady=pad, rowspan=2, sticky=align_mode)
div2.grid(column=1, row=0, padx=pad, pady=pad, sticky=align_mode)
div3.grid(column=1, row=1, padx=pad, pady=pad, sticky=align_mode)

define_layout(window, cols=2, rows=2)
define_layout([div1, div2, div3])

im = Image.open(file_path)
imTK = ImageTk.PhotoImage(im.resize((img_size, img_size)))

image_main = tk.Label(div1, image=imTK)
image_main['height'] = img_size
image_main['width'] = img_size

image_main.grid(column=0, row=0, sticky=align_mode)

import_btn = tk.Button(div2, text='開啟檔案', bg='#FFD2D2', fg='black')
save_btn = tk.Button(div2, text="儲存檔案", bg='#FFD2D2', fg='black')
import_btn.grid(column=0, row=0, sticky=align_mode)
save_btn.grid(column=0, row=1, sticky=align_mode)

bt1 = tk.Button(div3, text='Button 1', bg='#FFB5B5', fg='black')
bt2 = tk.Button(div3, text='Button 2', bg='#FFB5B5', fg='black')
bt3 = tk.Button(div3, text='Button 3', bg='#FFB5B5', fg='black')
bt4 = tk.Button(div3, text='Button 4', bg='#FFB5B5', fg='black')
bt1.grid(column=0, row=0, sticky=align_mode)
bt2.grid(column=0, row=1, sticky=align_mode)
bt3.grid(column=0, row=2, sticky=align_mode)
bt4.grid(column=0, row=3, sticky=align_mode)

define_layout(window, cols=2, rows=2)
define_layout(div1)
define_layout(div2, rows=2)
define_layout(div3, rows=4)

file_path = import_btn.bind('<Button-1>', openfile)
image_main.bind('<Button-1>', callback)
window.mainloop()
