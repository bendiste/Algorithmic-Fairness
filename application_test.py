
#Created by eNAS
#Import GUI interface Libraries
from tkinter import*
import tkinter as tk
from tkinter import ttk
import pickle
import tkinter.messagebox
import tksheet
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
import time
from dissim import *
import os, sys
from tkinter import colorchooser
from tkinter import font
from tkinter import simpledialog
import win32print
import win32api
import csv
import io
import pandas as pd
from tkinter import filedialog, Label, Button, Entry, StringVar
from tkinter.filedialog import askopenfilename
# from PIL import ImageTk,Image 
# from tkinter import filedialog
# from tkinter.ttk import 
#############################Import other .py files ############################################
from deepcopy import *
# from Adult_implementation import *
# from COMPAS_implementation import *  
# from German_implementation import *
from implementation_functions import *
# from german_imp import *
import skfuzzy as fuzz
from prince import FAMD #Factor analysis of mixed data
'--------------------------------------------------------------------------------'
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
from prince import FAMD #Factor analysis of mixed data
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from table2 import *
#Creating the application main root
root = Tk()
root.title('Fairness System')
root.title('Resizable')
root.geometry("1000x800")
my_menu = Menu(root)
root.config(menu=my_menu)

# Use this command to hide the tabs, and then show them
def our_command():
    pass

def hide():
    my_notebook = hide(frame2)
    
def show():
    my_notebook.add(frame2)

'--------------------------------------------------------------------------------------------------------'
#Start creating the tabs
my_notebook = ttk.Notebook(root)
my_notebook.pack(pady = 15)

##################################
#Start designing the frames
frame0 = Frame(my_notebook, width = 2000, height = 2000)
frame1 = Frame(my_notebook, width = 2000, height = 2000)

frame2 = Frame(my_notebook, width = 2000, height = 2000)
frame3 = Frame(my_notebook, width = 2000, height = 2000)
frame5 = Frame(my_notebook, width = 2000, height = 2000)

frame4 = Frame(my_notebook, width = 2000, height = 2000)
frame6 = Frame(my_notebook, width = 2000, height = 2000)
frame7 = Frame(my_notebook, width = 2000, height = 2000)

frame1.pack(fill = "both", expand = 1)
frame2.pack(fill = "both", expand = 1)

my_notebook.add(frame0, text = "Welcome")
my_notebook.add(frame1, text = "Dataset Upload")

my_notebook.add(frame2, text = "Dataset Select")
my_notebook.add(frame3, text = "Explore Data")

my_notebook.add(frame5, text = "Classifications")
my_notebook.add(frame4, text = "Clustering")

# my_notebook.add(frame6, text = "Plotting")
my_notebook.add(frame7, text = "Results")


#Start frame1
myLabel1 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel2 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()

myLabel3 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel4 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel5 = Label(frame1, text = "In case you choose your own data, please enter the following input variables", font = ("calibri",14) ).pack()
myLabel6 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel7 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()

#Upload the CSV file option

def UploadAction(event=None):
    filename = filedialog.askopenfile(mode='r', filetypes=[('all files', '*csv')])
    print('Selected:', filename)
    # inputlabel = Label(frame1, text = "The file you uploaded is" + filename).pack()
    # myLabelprint = Label(frame1, text = "The file you uploaded is"+ filename).pack()
    return filename

button = tk.Button(frame1, text='Upload csv file', command=UploadAction)
button.pack()

# def open_file():
#     file_path = askopenfile(mode='r', filetypes=[('all files', '*csv')])
#     if file_path is not None:
#         pass

# def uploadFiles():
#     pb1 = Progressbar(
#         frame1, 
#         orient=HORIZONTAL, 
#         length=300, 
#         mode='determinate'
#         )
#     pb1.grid(row=4, columnspan=3, pady=20)
#     for i in range(5):
#         frame1.update_idletasks()
#         pb1['value'] += 20
#         time.sleep(1)
#     pb1.destroy()
#     Label(frame1, text='File Uploaded Successfully!', foreground='green').grid(row=4, columnspan=3, pady=10)
           
# adhar = Label(
#     frame1, 
#     text='Upload Your Dataset '
#     )
# # adhar.grid(row=0, column=0, padx=10)
# print(uploadFiles)
# adharbtn = Button(
#     frame1, 
#     text ='Choose File', 
#     command = lambda:open_file()
#     ).pack()
# # adharbtn.grid(row=0, column=1)
# myLabel131 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel132 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
# #this button is responsible about taking the uploaded file and pass it yo the system
# upld = Button(
#     frame1, 
#     text='Upload Files', 
#     command=uploadFiles
#     ).pack()
# # upld.grid(row=3, columnspan=3, pady=10)
myLabel8 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel8 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
#######################Upload file new code ############
myLabel10 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel11 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel12 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()

myLabel13 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel14 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel15 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()

myLabel16 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel17 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel18 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()

# myLabel19 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel20 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel21 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()

# myLabel22= Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel23 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel24 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
     	
myLabel25 = Label(frame0, text = "Welcome to the Fairness system", font = ("Forte",40)  ).pack()

myLabel26 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel27 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel28 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()

myLabel29 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel30 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel31 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
   
myLabel32 = Label(frame0, text = "Masters Student:", font = ("calibri",14) ).pack()
myLabel33 = Label(frame0, text = "Begum Hattatoglu", font = ("calibri",12) ).pack()

myLabel34 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel35 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()

myLabel36 = Label(frame0, text = "Supervised by:", font = ("calibri",14)).pack()
myLabel37 = Label(frame0, text = "Hakim Qahtan", font = ("calibri",12) ).pack()

myLabel38 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()
myLabel39 = Label(frame0, text = "                                                                                                                                                                                                                                         ").pack()

myLabel40 = Label(frame0, text = "Copyrights", font = ("calibri",11) ).pack()
myLabel41 = Label(frame0, text = " @Utrecht University", font = ("calibri",11) ).pack()

# canvas = Canvas(frame0, width = 2000, height =800)  
# canvas.pack()  
# img = ImageTk.PhotoImage(Image.open("C:/Users/Khwai001/Fairness/Project Codes/Implementation/GUI_test/assets/Fairness_Logo_scale.jpg"))  
# canvas.create_image(5, 5, anchor=NW, image=img)

#############################################
# myLabel1 = Label(frame1, text = "Upload your data here").pack()

# e = Entry(frame1, width = 55)
# e.pack()
# e.insert(0,  "")

# Upload = Button(frame1, text = "Upload")
# Upload.pack()
#############################################
# myLabel1 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel1 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
input1 = Label(frame1, text = "Sensitive attribute 1", font = ("calibri",10) ).pack()
# input2 = Label(frame1, text = "Input is in list format").pack()
e1 = Entry(frame1, width = 10)
e1.pack()
e1.insert(0,"")
####################
input8 = Label(frame1, text = "Sensitive attribute 2", font = ("calibri",10) ).pack()
# input8 = Label(frame1, text = "Input is in list format").pack()
e5 = Entry(frame1, width = 10)
e5.pack()
e1.insert(0,"")
myLabel42 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel43 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
####################
input3 = Label(frame1, text = "Decision variable" ).pack()
input4 = Label(frame1, text = "String input").pack()
e2= Entry(frame1, width = 20)
e2.pack()
e2.insert(0,  "")
myLabel44 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel45 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
####################
input5 = Label(frame1, text = "Favorable label for sensetive attribute 1" ).pack()
input6 = Label(frame1, text = "Binary (0, 1)").pack()
e3= Entry(frame1, width = 20)
e3.pack()
e3.insert(0,  "")
myLabel46 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel47 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()

####################
input7 = Label(frame1, text = "Favorable label for sensetive attribute 2" ).pack()
input8 = Label(frame1, text = "Binary (0, 1)").pack()
e4= Entry(frame1, width = 20  )
e4.pack()
e4.insert(0,  "")
myLabel7 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel8 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()

proceed = Button(frame1, text = "Proceed" )
proceed.pack()

###### End of frame 1 ######
######################################################################################################
#Connecting to the data

###### Start of frame 2 ######
#create the list box to take the dataset
# listbox = Listbox(frame2)
# listbox.pack(pady = 70)

# listbox.insert(END, "German Dataset")
# listbox.insert(END, "Compas Dataset")
# listbox.insert(END, "Adult Dataset")

# def headclick():
#     # if datachoice == 1:
#         # german = pd.read_csv ('german.csv')
#         # print(frame2, german)
#         with open('german.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                tkinter.messagebox.showinfo(frame2, print(row))
    # elif datachoice == 2:
    #     compas = pd.read_csv ('compas-scores-two-years_original.csv')
    #     print (frame2,compas)
    # elif datachoice == 3:
    #     adult = pd.read_csv ('adult.csv')
    #     print (frame2, adult)

myLabel1 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()

def delete():
    listbox.delete(ANCHOR)
    
def select():
    label1.config(text = listbox.get(ANCHOR))

global label1

# alldata = np.vstack((X_train_reduc[0], X_train_reduc[1]))
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()

datachoice  = IntVar()

label1 = Label(frame2, text = 'Choose one of the datasets', font = ("calibri",18))
label1.pack(pady= 10) 
#Drop Down menu:
clicked = StringVar()
clicked.set("Datasets")

# myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
 
myLabel1 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame1, text = "                                                                                                                                                                                                                                         ").pack()

 
def germanshow():
    dataset_orig, privileged_groups, unprivileged_groups = aif_data("german", True)
    sens_attr = ['age', 'sex']
    decision_label = 'credit'
    fav_l = 1
    unfav_l = 0
    orig_df, num_list, cat_list = preprocess(dataset_orig, sens_attr, decision_label)
    # _, pg, upg, ds_orig = data_load("German")
    # orig_df = mypreprocess(ds_orig)
    tot_rows = 5
    tot_cols = len(orig_df.columns)
    h_data = orig_df.head(5)
    win = tk.Toplevel()
    h = Scrollbar(win, orient = 'horizontal')
    win.wm_title("German Dataset Head")
    t = Table(win, h_data, tot_rows, tot_cols)
    l = tk.Label(win, text = t)
    l.grid(row=0, column=0)
    # return  orig_df, privileged_groups, unprivileged_groups

def adultshow():
    dataset_orig, privileged_groups, unprivileged_groups = aif_data("adult", True)
    # orig = dataset_orig.convert_to_dataframe()
    # print(orig[0].columns)
    sens_attr = ['race', 'sex']
    decision_label = 'Income Binary'
    fav_l = 1
    unfav_l = 0
    orig_df, num_list, cat_list = preprocess(dataset_orig, sens_attr, decision_label)
    # print(orig_df)
    tot_rows = 5
    tot_cols = len(orig_df.columns)
    h_data = orig_df.head(5)
    win = tk.Toplevel()
    win.wm_title("Adult Dataset Head")
    t = Table(win, h_data, tot_rows, tot_cols)
    l = tk.Label(win, text = t)
    l.grid(row=0, column=0)
    return orig_df, privileged_groups, unprivileged_groups
    
def compasshow():
    dataset_orig, privileged_groups, unprivileged_groups = aif_data("compas", True)
    sens_attr = ['race', 'sex']
    decision_label = 'two_year_recid'
    fav_l = 1
    unfav_l = 0
    orig_df, num_list, cat_list = preprocess(dataset_orig, sens_attr, decision_label)
    tot_rows = 5
    tot_cols = len(orig_df.columns)
    h_data = orig_df.head(5)
    win = tk.Toplevel()
    win.wm_title("Compas Dataset Head")
    t = Table(win, h_data, tot_rows, tot_cols)
    l = tk.Label(win, text = t)
    l.grid(row=0, column=0)
    return orig_df, privileged_groups, unprivileged_groups

germanbutton = Radiobutton(frame2, text = "German Dataset",variable = datachoice, value = 1, command = germanshow).pack()
adultbutton = Radiobutton(frame2, text = "Adult Dataset",variable = datachoice, value = 2, command = adultshow).pack()
compasbutton = Radiobutton(frame2, text = "Compas Dataset",variable = datachoice, value = 3, command = compasshow).pack()

myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
    
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()

myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame2, text = "                                                                                                                                                                                                                                         ").pack()
# # frame2.add_separator()
# listbutton = Button(frame2, text="Delete")
# listbutton.pack(pady = 30, padx = 40)

# previuosbutton = Button(frame2, text="<<Previous", command = select)
# previuosbutton.pack(pady = 30, padx =10)

# nextbutton = Button(frame2, text="Next>>")
# nextbutton.pack(pady = 30, padx = 70)

###### End of frame 2 ######

###### Start of frame 3 ######

myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()

myLabel1 = Label(frame3, text = "Start Exploring your dataset", font = ("calibri",18) ).pack()

myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()

def compute_initial_DI_Ratio(dataset = 'German'):
    dataset_orig, privileged_groups, unprivileged_groups = aif_data(dataset, False)
    metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
    return metric_orig, privileged_groups, unprivileged_groups

# def data_load(data_name = "german"):    
#     di, privileged_groups, unprivileged_groups, ds_orig = compute_initial_DI_Ratio(data_name)
#     dataset_orig, privileged_groups, unprivileged_groups = aif_data(data_name, False)
#     return di, privileged_groups, unprivileged_groups, ds_orig
# disp_imp, pg, upg, ds_orig = data_load()  

def impratio():
    data_name = "compas"
    di, privileged_groups, unprivileged_groups = compute_initial_DI_Ratio(data_name)
    win = tk.Toplevel()
    win.wm_title("Impact ratio")
    # label1 = tk.Label("Disparate impact (of original labels) between unprivileged and privileged groups = %f")
    l = tk.Label(win, text = di.disparate_impact())
    # label11 = tk.Label("Privileged_groups=")
    m = tk.Label(win, text = privileged_groups)
    # label12 = tk.Label("Unprivileged_groups=")
    m2 = tk.Label(win, text = unprivileged_groups)
    row_nr = 0
    # label1.grid(row=row_nr, column=0)
    # row_nr += 1
    l.grid(row=row_nr, column=0)
    row_nr += 1
    # label11.grid(row=row_nr, column=0)
    # row_nr += 1
    m.grid(row=row_nr, column=0)
    row_nr += 1
    # label12.grid(row=row_nr, column=0)
    # row_nr += 1
    m2.grid(row=row_nr, column=0)
    row_nr += 1
    b = ttk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=row_nr, column=0)
    
ratio = Button(frame3, text = "Disparate Impact ratio", command = impratio)
ratio.pack()

myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()

def demgraphic():
    data_name = "compas"
    di, privileged_groups, unprivileged_groups = compute_initial_DI_Ratio(data_name)
    win = tk.Toplevel()
    win.wm_title("Demographic Parity Difference")
    l = tk.Label(win, text = di.statistical_parity_difference())
    row_nr = 0
    # label2.grid(row=row_nr, column=0)
    # row_nr += 1
    l.grid(row=row_nr, column=0)
    row_nr += 1
    b = ttk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=row_nr, column=0)
Demo = Button(frame3, text = "Demographic Parity Difference", command =demgraphic)
Demo.pack()

myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()

def consistency():
    data_name = "german"
    di, privileged_groups, unprivileged_groups = compute_initial_DI_Ratio(data_name)
    win = tk.Toplevel()
    win.wm_title("Consistency")
    l = tk.Label(win, text = di.consistency())
    row_nr = 0
    l.grid(row=row_nr, column=1)
    row_nr += 1
    b = ttk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=row_nr, column=1)
Consistency = Button(frame3, text = "Consistency", command =consistency)
Consistency.pack()

myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame3, text = "                                                                                                                                                                                                                                         ").pack()

def info():
    win = tk.Toplevel()
    win.wm_title("Information")
    l = tk.Label(win, text="PLEASE ADD THE DESCRIPTION HERE ")
    l.grid(row=0, column=0)

    b = ttk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=1, column=0)

info6 = Button(frame3, text = "Read more!", command = info )
info6.pack()
###### End of frame3 ######

###### Start of frame 4 ######
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()

myLabel1 = Label(frame4, text = "Clustering Methods", font = ("calibri",21)).pack()

myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()

myLabel14 = Label(frame4, text = "Fuzzy C-means plottings", font = ("calibri",16)).pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()


def fuzzy_2d_plot():
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    fpcs = []
    
    #checking for the optimal num of clusters with FPC plots
    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    
        # Store fpc values for later
        fpcs.append(fpc)
    
        # Plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(u, axis=0)
        for j in range(ncenters):
            ax.plot(X_train_reduc[0][cluster_membership == j],
                    X_train_reduc[1][cluster_membership == j], '.', color=colors[j])
    
        # Mark the center of each fuzzy cluster
        for pt in cntr:
            ax.plot(pt[0], pt[1], 'rs')
    
        ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
        ax.axis('off')
    fig1.tight_layout()
    
    #fpc plot per number of clusters
    fig2, ax2 = plt.subplots()
    ax2.plot(np.r_[2:11], fpcs)
    ax2.set_xlabel("Number of centers")
    ax2.set_ylabel("Fuzzy partition coefficient")
    
    return fig1,  fig2

def twoDplots():
    win = tk.Toplevel()
    win.wm_title("Two Dimensional Plotting")
    l = tk.Label(win, text = fuzzy_2d_plot())
    l.grid(row=0, column=0)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=0)
    return

head = Button(frame4, text = "2D plottings", command = twoDplots)
head.pack()

myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()

# def cluster(n_clust, X_train_reduc):
#     cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, n_clust, 2, error=0.005, 
#                                                       maxiter=5000)
#     # u: final fuzzy-partitioned matrix, u0: initial guess at fuzzy c-partitioned matrix,
#     # d: final euclidean distance matrix, jm: obj func hist, p: num of iter run, 
#     #fpc: fuzzy partition coefficient
#     u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
#                                                        maxiter=5000)
#     clusters = np.argmax(u, axis=0)  # Hardening for silhouette
#     return clusters, cntr

def silhouette_plot():
    from tqdm import tqdm
    n_clusters = []
    silhouette_scores = []
    for i in tqdm(range(2, 10)):
        try:
            cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, i, 2, error=0.005, 
                                                              maxiter=5000)
            u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
                                                                maxiter=5000)
            clusters = np.argmax(u, axis=0)
            silhouette_val = silhouette_score(X_train_reduc, clusters, 
                                              metric='euclidean')
            silhouette_scores.append(silhouette_val)
            n_clusters.append(i)
        except:
            print(f"Can't cluster with {i} clusters")
    plt.scatter(x=n_clusters, y=silhouette_scores)
    plt.plot(n_clusters, silhouette_scores)
    plt.show()

def silhouette():
    win = tk.Toplevel()
    win.wm_title("Silhouette")
    l = tk.Label(win, text = silhouette_plot())
    l.grid(row=0, column=0)
    return

myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()

head = Button(frame4, text = "Silhouette plot", command = silhouette)
head.pack()

myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()

def numofclustrers():
    clf = e6.pack()
    win = tk.Toplevel()
    win.wm_title("Number of Clusters")
    l = tk.Label(win, text = "Number of Clusters")
    l.grid(row=0, column=3)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=3)
    return


head = Button(frame4, text = "Number of clusters plot", command = numofclustrers )
head.pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()

input6 = Label(frame4, text = "Input number of clusters").pack()
e6 = Entry(frame4, width = 10)
e6.pack()
'-------------------------------------------------------------------------'
# clf = e6.get()   
# clf = RandomForestClassifier()
'---------------------------------------------------'
def cluster(n_clust, X_train_reduc):
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, n_clust, 2, error=0.005, 
                                                      maxiter=5000)
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
                                                       maxiter=5000)
    clusters = np.argmax(u, axis=0)  # Hardening for silhouette
    return clusters, cntr
'---------------------------------------------------'

def clusnum():
    win = tk.Toplevel()
    win.wm_title("Two Dimensional Plotting")
    l = tk.Label(win, text = "Output to number of clusters")
    l.grid(row=0, column=0)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=0)   
    return

myLabel1 = Label(frame4, text = "                                                                                                                                                                                                                                         ").pack()
head = Button(frame4, text = "Take clusters number", command = clusnum)
head.pack()

###### End of frame 4 ######

###### Start of frame 5 ######
myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()

myLabel1 = Label(frame5, text = "Choose the classifications algorithm", font = ("calibri",21) ).pack()

def LR():

    return 

def LRtwoDplots():
    win = tk.Toplevel()
    win.wm_title("Two Dimensional Plotting")
    l = tk.Label(win, text ="Logistic Regression Plotting ")
    l.grid(row=0, column=0)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=0)
    return 

head = Button(frame5, text = "Logistic Regression", command = LRtwoDplots)
head.pack()

myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()

def RF(): 
    clf = RandomForestClassifier()
    # NOTE: clf must come from the user!
    # Getting the baseline performance results from the imbalanced dataset
    # Note: the function is created based on the assumption that the X's have sub_labels
    # Instantiate the desired classifier obj to train the classification models    
    baseline_stats, cm, ratio_table, preds = baseline_metrics(clf, X_train, X_test, 
                                                  y_train, y_test, sens_attr, 
                                                  fav_l, unfav_l)
    return

def RFtwoDplots():
    win = tk.Toplevel()
    win.wm_title("Two Dimensional Plotting")
    l = tk.Label(win, text ="Random Forest Plotting ")
    l.grid(row=0, column=0)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=0)
    return 

head = Button(frame5, text = "Random Forest", command = RFtwoDplots)
head.pack()

myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()

def GB():
    clf = GradientBoostingClassifier()
    baseline_stats, cm, ratio_table, preds = baseline_metrics(clf, X_train, X_test, 
                                                  y_train, y_test, sens_attr, 
                                                  fav_l, unfav_l)
    return baseline_stats, cm, ratio_table

def GBtwoDplots():
    win = tk.Toplevel()
    win.wm_title("Two Dimensional Plotting")

    l = tk.Label(win, text ="Gradient Boosting Plotting ")
    l.grid(row=0, column=1)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=1)
    return 
    

head = Button(frame5, text = "Gradient Boosting", command = GBtwoDplots)
head.pack()

myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()
# myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()
myLabelresults = Label(frame5, text = "The Baseline Statistics", font = ("calibri",21) ).pack()

def fairnessmatrix():
    win = tk.Toplevel()
    win.wm_title("Fairness Matrix")
    l = tk.Label(win, text ="Fairness Matrix")
    l.grid(row=0, column=2)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=2)
    return cm

head = Button(frame5, text = "Fairness and performance Metrics", command = fairnessmatrix)
head.pack()

myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()

def commatrix():
    # cm = classified_metric.binary_confusion_matrix()
    win = tk.Toplevel()
    win.wm_title("Binary Confusion Matrix")
    l = tk.Label(win, text ="Binary Confusion Matrix")
    l.grid(row=0, column=1)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=1)
    return cm

head = Button(frame5, text = "Confusion Matrix", command = commatrix)
head.pack()

myLabel1 = Label(frame5, text = "                                                                                                                                                                                                                                         ").pack()

def ratios():
    win = tk.Toplevel()
    win.wm_title("Ratio Matrix")
    l = tk.Label(win, text ="Ratios matrix")
    l.grid(row=0, column=1)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=1)
    return

head = Button(frame5, text = "Ratios", command = ratios)
head.pack()
###### End of frame 5 ######

###### Start of frame 6 ######

###### End of frame 6 ######

###### Start of frame 7 ######
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()

myLabel1 = Label(frame7, text = "Fairness and performance metrics tables", font = ("calibri",21) ).pack()

myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()

def metrics_strategy1(X_test, X_test_pred1, y_test, sens_attr, fav_l, unfav_l):
    metrics_table1, cm1, ratio_t1 = metrics_calculate(X_test, X_test_pred1, y_test,
                                                  sens_attr, fav_l, unfav_l)
    return metrics_table1, cm1, ratio_t1

def m1():
    win = tk.Toplevel()
    win.wm_title("Strategy 1")
    l = tk.Label(win, text ="Metric Table: " )
    l1 = tk.Label(win, text = "Confusion Matrix: " )
    l2 = tk.Label(win, text = "Ratio Table: " )
    l.grid(row=0, column=3)
    l1.grid(row=1, column=3)
    l2.grid(row=2, column=3)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=3)
    return
    
myLabel14 = Label(frame7, text = "Confusion matrix", font = ("calibri",14) ).pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()

head = Button(frame7, text = "Strategy 1", command =  m1)
head.pack()

myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()

def metrics_strategy1(X_test, X_test_pred2, y_test, sens_attr, fav_l, unfav_l):
    metrics_table2, cm2, ratio_t2 = metrics_calculate(X_test, X_test_pred2, y_test,
                                                  sens_attr, fav_l, unfav_l)
    return metrics_table2, cm2, ratio_t2

def m2():
    win = tk.Toplevel()
    win.wm_title("Strategy 2")
    l = tk.Label(win, text ="Metric Table: "  )
    l1 = tk.Label(win, text = "Confusion Matrix: "  )
    l2 = tk.Label(win, text = "Ratio Table: "  )
    l.grid(row=0, column=3)
    l1.grid(row=1, column=3)
    l2.grid(row=2, column=3)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=3)
    return
    
myLabel14 = Label(frame7, text = "Subgroup ratio table", font = ("calibri",14) ).pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
head = Button(frame7, text = "Strategy 2",  command = m2)
head.pack()

myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()

def metrics_strategy1(X_test, X_test_pred3, y_test, sens_attr, fav_l, unfav_l):
    metrics_table3, cm3, ratio_t3 = metrics_calculate(X_test, X_test_pred3, y_test,
                                                  sens_attr, fav_l, unfav_l)
    return metrics_table3, cm3, ratio_t3

def m3():
    win = tk.Toplevel()
    win.wm_title("Strategy 3")
    l = tk.Label(win, text ="Metric Table: " )
    l1 = tk.Label(win, text = "Confusion Matrix:" )
    l2 = tk.Label(win, text = "Ratio Table: " )
    l.grid(row=0, column=3)
    l1.grid(row=1, column=3)
    l2.grid(row=2, column=3)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=3, column=3)
    return
    
myLabel14 = Label(frame7, text = "Gradient Boosting Trees", font = ("calibri",14) ).pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
head = Button(frame7, text = "Strategy 3",  command =  m3)
head.pack()

myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()
myLabel1 = Label(frame7, text = "                                                                                                                                                                                                                                         ").pack()

###### End of frame 7 ######

#############################
#Bulding up the menu
#############################    
file_menu = Menu(my_menu)
#############################
my_menu.add_cascade(label = "File", menu= file_menu)  #new bar tick
file_menu.add_command(label="New", command = file_menu)    #new bar tick
file_menu.add_separator()
file_menu.add_command(label="Exit", command = root.quit)
##############################
edit_menu = Menu(my_menu)
my_menu.add_cascade(label= "Edit", menu = edit_menu) #new bar tick
edit_menu.add_command(label = "Cut", command = our_command())
edit_menu.add_separator()
edit_menu.add_command(label = "Copy", command = our_command())
##############################
my_menu.add_cascade(label = "Help")
##############################
options_menu = Menu(my_menu)
my_menu.add_cascade(label= "Options", menu = options_menu)   #new bar tick
options_menu.add_command(label = "Find", command = our_command())
options_menu.add_separator()
options_menu.add_command(label = "Find next", command = our_command())

root.mainloop()