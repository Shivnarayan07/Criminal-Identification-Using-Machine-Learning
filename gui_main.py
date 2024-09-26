import tkinter  as tk 
from tkinter import * 
import time
import numpy as np

import os
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 
from PIL import Image , ImageTk  

root = tk.Tk()
#root.geometry('500x500')
#root.title("Login Form")


#------------------------------------------------------

root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Crime Prediction and Criminal Identification Using Machine Learning")
#------------------Frame----------------------



#-------function------------------------

def crime():
    
##### tkinter window ######
    
    print("reg")
    from subprocess import call
    call(["python", "registration.py"])   



def criminal():
   

    from subprocess import call
    call(["python", "GUI_master.py"])   
    


#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('Slide2.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)


lbl = tk.Label(root, text="Crime Prediction and Criminal Identification using Machine Learning", font=('times', 40,' bold '), height=1, width=50,bg="gold",fg="white")
lbl.place(x=0, y=5)

framed = tk.LabelFrame(root, text=" --WELCOME-- ", width=500, height=250, bd=5, font=('times', 14, ' bold '),bg="gold")
framed.grid(row=0, column=0, sticky='nw')
framed.place(x=450, y=300)
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
button1 = tk.Button(framed, text='Crime Prediction ',width=20,height=3,bg='dark blue',fg='white',command=crime,font='bold').place(x=30,y=50)
button1 = tk.Button(framed, text='Criminal Identification',width=20,height=3,bg='dark blue',fg='white',command=criminal,font='bold').place(x=280,y=50)


root.mainloop()