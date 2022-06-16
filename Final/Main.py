import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import dlib
from MLS_Deformation import * 

# (對68個點要變動的程式碼，每對imgS做變動，都要呼叫Renew()函式更新)


def setting_scale():
    eyescale1.set(0) 
    eyescale2.set(0) 
    eyescale3.set(0) 
    facescale1.set(0) 
    nosescale1.set(0) 
    nosescale2.set(0) 
    mouthscale1.set(0) 
    mouthscale2.set(0) 

def OpenCV_small():  # opencv 等比縮小圖片
    global imgS    
    nr,nc = imgS.shape[:2]
    if nr >= 800 or nc >= 800:
        if nr>= nc:
            scale = 800/nr 
        else:
            scale = 800/nc
        nr2 = int(nr * scale)
        nc2 = int(nc * scale)
        imgS = cv2.resize(imgS,(nc2,nr2),interpolation = cv2.INTER_LINEAR)


def open_file():
    global panel,imgS,img_show,imgO,imgN
    filename=filedialog.askopenfilename()  #獲取文件全路徑
    imgS = cv2.imread(filename) # 用opencv的方法   
    imgN = imgS.copy()
    OpenCV_small() # opencv 等比縮小圖片
    
    # Label改成Canvas，不要用Label因為會有覆寫問題
    panel = tk.Canvas(block1,width=650,height=800,bg='#FFFFFF') # 設定長寬   
    imgS,landmarks = rection(imgS)  # imgS是rection後的opencv圖    
    Renew(imgS) # 更新
    
    setting_scale()
    
def save_file():
    global new_img,img_show,imgS
    file = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=(("PNG file", "*.png"),("All Files", "*.*") ))
    cv2.imwrite(file.name,imgS)

    

def Renew(im):
    # 把img_show變成PIL格式圖
    global imgS,img_show,panel,new_img
    img_show = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    new_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_show = Image.fromarray(img_show)
    img_show = ImageTk.PhotoImage(image = img_show)
    # 要更新
    panel.delete("all") 
    panel.create_image(0,0,image=img_show,anchor=NW)
    panel.place(x=0,y=0)

def rection(img): # OpenCV 測68個點
    #dlib預測器
    global landmarks
    detector = dlib.get_frontal_face_detector()    #使用dlib庫提供的人臉提取器
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   #構建特徵提取器
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人臉數rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])  #人臉關鍵點識別
        for idx, point in enumerate(landmarks):        #enumerate函式遍歷序列中的元素及它們的下標
            # 利用cv2.circle給每個特徵點畫一個圈，共68個
            pos = (point[0, 0], point[0, 1])
            #cv2.circle(img, pos, 4, color=(0, 255, 0),thickness = -1)
        
    return img,landmarks

global enlarge_value0
    
def get_value(event,num):
    global enlarge_value,imgS,enlarge_value0,imgN
    #try:
    if(num==1):
        enlarge_value = eyescale1.get() - enlarge_value0
        print(enlarge_value)
    elif(num==2):
        enlarge_value = eyescale2.get() - enlarge_value0
        print(enlarge_value)
    elif(num==3):
        enlarge_value = eyescale3.get() - enlarge_value0
        print(enlarge_value)
    elif(num==4):
        enlarge_value = facescale1.get() - enlarge_value0
        print(enlarge_value)
    elif(num==5):
        enlarge_value = nosescale1.get() - enlarge_value0
        print(enlarge_value)
    elif(num==6):
        enlarge_value = nosescale2.get() - enlarge_value0
        print(enlarge_value)
    elif(num==7):
        enlarge_value = mouthscale1.get() - enlarge_value0
        print(enlarge_value)
    elif(num==8):
        enlarge_value = mouthscale2.get()
        print(enlarge_value)
    img,landmarks = rection(imgS)
    imgS,imgN = eye_deformation(landmarks,imgS,num,enlarge_value,imgN)
    #imgN = eye_deformation(landmarks,imgS,num,enlarge_value)
    Renew(imgS)
    #except:
        #print('請選擇照片')
        
def get_value0(event,num):
    global enlarge_value0
    if(num==1):
        enlarge_value0 = eyescale1.get()
    elif(num==2):
        enlarge_value0 = eyescale2.get()
    elif(num==3):
        enlarge_value0 = eyescale3.get()
    elif(num==4):
        enlarge_value0 = facescale1.get()
    elif(num==5):
        enlarge_value0 = nosescale1.get()
    elif(num==6):
        enlarge_value0 = nosescale2.get()
    elif(num==7):
        enlarge_value0 = mouthscale1.get()
    elif(num==8):
        enlarge_value0 = mouthscale2.get()
        



global imgS,img_show,panel,enlarge_value0,new_img,imgO,imgN

   
window = tk.Tk()
window.title('人臉五官微調系統')
window.geometry('960x800')

div_size = 300
align_mode = 'nswe'
pad0 = 10
pad = 5

#GUI整體布局
blocktop = tk.Frame(window, width=650, height=30)
blocktop2 = tk.Frame(window, width=div_size, height=30)
block1 = tk.Frame(window, width=650, height=800,bg='#FFFFFF')
block2 = tk.Frame(window, width=div_size, height=250)
block3 = tk.Frame(window, width=div_size, height=250)
block4 = tk.Frame(window, width=div_size, height=250)
block5 = tk.Frame(window, width=div_size, height=250)

blocktop.grid(column=0, row=0, sticky=align_mode)
blocktop2.grid(column=1, row=0, sticky=align_mode)
block1.grid(column=0, row=1, rowspan=5, sticky=align_mode) 
block2.grid(column=1, row=2, sticky=align_mode)
block3.grid(column=1, row=3, sticky=align_mode)
block4.grid(column=1, row=4, sticky=align_mode)
block5.grid(column=1, row=5, sticky=align_mode)


#拉霸區塊布局
eyelabel = tk.Label(block2,text="眼睛",font=('新細明體', 12),padx=pad, pady=pad,fg='#007799')
font = ('Courier New', 20, 'bold')
eyescale1 = tk.Scale(
    block2, label='大小', from_=-10, to=10, orient="horizontal"
    ,tickinterval=5,length=280)
eyescale1.bind('<Button-1>', lambda event: get_value0(event, 1)) 
eyescale1.bind('<ButtonRelease-1>', lambda event: get_value(event, 1)) 

eyescale2 = tk.Scale(
    block2, label='眼高', from_=-10, to=10, orient="horizontal"
    ,tickinterval=5,length=280)
eyescale2.bind('<Button-1>', lambda event: get_value0(event, 2))
eyescale2.bind('<ButtonRelease-1>', lambda event: get_value(event, 2)) 

eyescale3 = tk.Scale(
    block2, label='眼距', from_=-10, to=10, orient="horizontal"
    ,tickinterval=5,length=280)
eyescale3.bind('<Button-1>', lambda event: get_value0(event, 3))
eyescale3.bind('<ButtonRelease-1>', lambda event: get_value(event, 3)) 

eyelabel.grid(row=0, column=0)
eyescale1.grid(row=1, column=0)
eyescale2.grid(row=2, column=0)
eyescale3.grid(row=3, column=0)

noselabel = tk.Label(block3,text="鼻子",font=('新細明體', 12),padx=pad, pady=pad,fg='#007799')
nosescale1 = tk.Scale(
    block3, label='大小', from_=-8, to=8, orient="horizontal",tickinterval=4,length=280)
nosescale2 = tk.Scale(
    block3, label='鼻翼', from_=-20, to=20, orient="horizontal",tickinterval=5,length=280)
nosescale1.bind('<Button-1>', lambda event: get_value0(event, 5))
nosescale1.bind('<ButtonRelease-1>', lambda event: get_value(event, 5)) 

nosescale2.bind('<Button-1>', lambda event: get_value0(event, 6))
nosescale2.bind('<ButtonRelease-1>', lambda event: get_value(event, 6)) 


noselabel.grid(row=0, column=0)
nosescale1.grid(row=1, column=0)
nosescale2.grid(row=2, column=0)

mouthlabel = tk.Label(block4,text="嘴巴",font=('新細明體', 12),padx=pad, pady=pad,fg='#007799')
mouthscale1 = tk.Scale(
    block4, label='大小', from_=-10, to=10, orient="horizontal",tickinterval=5,length=280)
mouthscale2 = tk.Scale(
    block4, label='顏色', from_=0, to=255, orient="horizontal",tickinterval=50,length=280)

mouthscale1.bind('<Button-1>', lambda event: get_value0(event, 7))
mouthscale1.bind('<ButtonRelease-1>', lambda event: get_value(event, 7)) 

mouthscale2.bind('<Button-1>', lambda event: get_value0(event, 8))
mouthscale2.bind('<ButtonRelease-1>', lambda event: get_value(event, 8)) 


mouthlabel.grid(row=0, column=0)
mouthscale1.grid(row=1, column=0)
mouthscale2.grid(row=2, column=0)

facelabel = tk.Label(block5,text="臉型",font=('新細明體', 12),padx=pad, pady=pad,fg='#007799')
font = ('Courier New', 20, 'bold')
facescale1 = tk.Scale(
    block5, label='瘦臉', from_=-15, to=15, orient="horizontal",tickinterval=5,length=280)

facelabel.grid(row=0, column=0)
facescale1.grid(row=1, column=0)
facescale1.bind('<Button-1>', lambda event: get_value0(event, 4)) 
facescale1.bind('<ButtonRelease-1>', lambda event: get_value(event, 4)) 

#開檔&存檔
import_btn = tk.Button(blocktop, text='開啟檔案', bg='#BBFFEE', fg='black',height = 1, 
          width = 20, command=open_file)
save_btn = tk.Button(blocktop, text="儲存檔案", bg='#BBFFEE', fg='black',height = 1, 
          width = 20, command=save_file)
import_btn.grid(column=0, row=0, padx=pad0, pady=pad0, sticky=align_mode)
save_btn.grid(column=1, row=0, padx=pad0, pady=pad0, sticky=align_mode)

#顯示照片
mainImage=tk.Label(block1,height = 1, width = 95, image=None,bg='#FFFFFF') 
mainImage.grid(row=0, column=0, sticky=align_mode)


window.mainloop()