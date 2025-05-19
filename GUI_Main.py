import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
import cv2

##############################################+=============================================================
root = tk.Tk()
root.configure(background="skyblue")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Twitter Sentiment Analysis Using Machine Learning")


# bg = Image.open(r"D:\GUIcode\image4.jpg")

# # bg.resize((1366,500),Image.ANTIALIAS)
# # print(w,h)
# bg_img = ImageTk.PhotoImage(bg)
# bg_lbl = tk.Label(root, image=bg_img)
# bg_lbl.place(x=0, y=93, relwidth=1, relheight=1)

'''
bg = PhotoImage(file="image3.jpg")
label1 = Label(root, image=bg)
label1.place(x=0, y=0)
'''


img1 = ImageTk.PhotoImage(Image.open("a1.png"))

img2 = ImageTk.PhotoImage(Image.open("a11.jpeg"))

img3 = ImageTk.PhotoImage(Image.open("a1.jpg"))

#img4 = ImageTk.PhotoImage(Image.open("img7.jpg"))

logo_label = tk.Label()
logo_label.place(x=60, y=150)

x = 1


def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img1)
	elif x == 2:
		logo_label.config(image=img2)
	elif x == 3:
		logo_label.config(image=img3)
     # elif x ==4:
     #    logo_label.config(image=img4)
	x = x+1
	root.after(1000, move)

# calling the function
move()

image2 =Image.open('twit.png')
image2 =image2.resize((100,80), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image,bd=5)

background_label.image = background_image

background_label.place(x=1090, y=80)

w = tk.Label(root, text="_Twitter Sentiment Analysis Using Machine Learning_",width=40,background="skyblue",height=2,font=("Times new roman",24,"bold"))
w.place(x=80,y=15)

# lbl = tk.Label(root, text="Twitter Data Analysis Using Machine Learning", font=('Times', 35,' bold '),bg="black",fg="white")
# lbl.place(x=20, y=12)




def login():

    from subprocess import call
    call(["python", "login.py"])  

def register():

    from subprocess import call
    call(["python", "registration.py"])  
   
def window():
    root.destroy()


button1 = tk.Button(root, text=" SIGN UP ",command=register,width=15, height=3,bd=5, font=('times', 17, ' bold '),bg="gray",fg="white")
button1.place(x=1040, y=250)

button2 = tk.Button(root, text="LOGIN",command=login,width=15, height=3,bd=5, font=('times', 17, ' bold '),bg="gray",fg="white")
button2.place(x=1040, y=450)




root.mainloop()
