import pandas as pd, numpy as np, re
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################
    
root = tk.Tk()
root.title("Twitter Data Analysis Using Machine Learning")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

image2 =Image.open(r'bgg.webp')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

image2 =Image.open(r's1.png')
image2 =image2.resize((1000,500), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=500, y=170)

###########################################################################################################
# lbl = tk.Label(root, text="Twitter Data Analysis Using Machine Learning", font=('times', 35,' bold '), height=1, bg="#FFBF40",fg="black")
# lbl.place(x=300, y=10)
##############################################################################################################################


def Data_Display():
    columns = ['id', 'tweet', 'target']
    print(columns)

    data1 = pd.read_csv(r"tweets.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    ID = data1.iloc[:, 0]
    Tweet = data1.iloc[:, 1]
    Target = data1.iloc[:, 2]


    display = tk.LabelFrame(root, width=100, height=400, )
    display.place(x=600, y=100)

    tree = ttk.Treeview(display, columns=(
    'ID', 'Tweet', 'Target'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=40)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Calibri', 10), background="black")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3")
    tree.column("1", width=130)
    tree.column("2", width=150)
    tree.column("3", width=200)

    tree.heading("1", text="ID")
    tree.heading("2", text="Tweet")
    tree.heading("3", text="target")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 304):
        tree.insert("", 'end', values=(
        ID[i], Tweet[i], Target[i]))
        i = i + 1
        print(i)

##############################################################################################################


def Train():
    
    result = pd.read_csv(r"tweets.csv",encoding = 'unicode_escape')

    result.head()
        
    result['headline_without_stopwords'] = result['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    
    def pos(headline_without_stopwords):
        return TextBlob(headline_without_stopwords).tags
    
    
    os = result.headline_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()
    
    os1['pos'] = os1['headline_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    result_train, result_test, label_train, label_test = train_test_split(result['pos'], result['target'],
                                                                              test_size=0.2, random_state=1)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(result_train)
    X_test_tf = tf_vect.transform(result_test)
    
    #
    
    clf = svm.SVC(C=10, gamma=0.001, kernel='linear')   
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(result_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    
    dump (clf,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")



entry = tk.Entry(root,width=20,font=("Tempus Sanc ITC",20))
entry.insert(0,"Enter tweet here...")
entry.place(x=10,y=350)
##############################################################################################################################################################################
def Test():
    predictor = load("SVM_MODEL.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
    if y_predict[0]==0:
        label4 = tk.Label(root,text ="Positive Tweet",width=20,height=2,bg='#46C646',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=700)
    else:
        label4 = tk.Label(root,text ="Negative Tweet",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=700)
    
###########################################################################################################################################################
def window():
    root.destroy()
    
button1 = tk.Button(root,command=Data_Display,text="Data_Display",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",20,"bold"))
button1.place(x=25,y=250)

# button2 = tk.Button(root,command=Train,text="Train",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
# button2.place(x=25,y=100)

button3 = tk.Button(root,command=Test,text="Test",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",20,"bold"))
button3.place(x=25,y=450)

button4 = tk.Button(root,command=window,text="Exit",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",20,"bold"))
button4.place(x=25,y=550)




root.mainloop()