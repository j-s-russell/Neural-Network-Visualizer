from tkinter import *
from customtkinter import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import seaborn as sns
import time
from nn_model import *
import numpy as np

set_appearance_mode("dark")

root = CTk()
root.title("Neural Network Visualizer")
root.geometry("800x800")

frame1 = CTkFrame(master=root)
frame1.pack(pady=20, padx=40, side=LEFT, expand=True)

title_label = CTkLabel(text="Left Click for Blue Points, Right Click for Red Points", master=frame1)
title_label.pack(pady=12, padx=10)

frame2 = CTkFrame(master=root)
frame2.pack(pady=20, padx=40, side=RIGHT, expand=True)

title_label = CTkLabel(text="Watch it learn!", master=frame2)
title_label.pack(pady=12, padx=10)

points = []

colors = {0: "red", 1: "blue"}

def paint_red(event):
    if len(points) <= 11:
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        points.append([event.x, 300 - event.y, 0])
        canvas.create_oval(x1, y1, x2, y2, fill="red")
        print(event.x, event.y)

def paint_blue(event):
    if len(points) <= 11:
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        points.append([event.x, 300 - event.y, 1])
        canvas.create_oval(x1, y1, x2, y2, fill="blue")
        print(event.x, event.y)

def clear_canvas():
    canvas.delete("all")
    points = []




canvas = Canvas(frame1, width=295, height=295, bg="white")
canvas.pack()
canvas.bind("<ButtonPress-1>", paint_red)
canvas.bind("<ButtonPress-2>", paint_blue)

fig = plt.Figure(figsize=(3,3), dpi=100)
ax = fig.add_subplot(111)
ax.set_axis_off()
plotarea = FigureCanvasTkAgg(fig, master = frame2)
plotarea.draw()
plotarea.get_tk_widget().pack()



# TRAINING


layer_dims = [2, 8, 16, 8, 1]
learning_rate = 0.0075
iterations = 12000

def iter_params(model, x, y):
    l_params = []
    for i in range(200):
        model.partial_train(x, y, layer_dims, learning_rate, 400, print_cost=False)
        l_params.append(model.params)
    return l_params

def iter_Z(model, x, y, xx, yy):
    Z_list = []
    for i in range(200):
        model.partial_train(x, y, layer_dims, learning_rate, 400, print_cost=False)
        Z = predict(model.params, np.c_[xx.ravel(), yy.ravel()].T)
        Z = Z.reshape(xx.shape)
        Z_list.append(Z)
    return Z_list


def predict(params, X):
    m = X.shape[1]
    p = np.zeros((1,m))

    # Forward propagation
    probas, caches = forprop(X, params)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    return p

def train():
    df = pd.DataFrame(data=points, columns=["x", "y", "color"])

    df.x = df.x / 300
    df.y = df.y / 300
    x = np.array(df.drop(["color"], axis=1))
    y = np.array(df["color"])

    # Reshape
    x = x.T
    # Add second dimension
    y = y.reshape(1, len(points))


    x_min, x_max, y_min, y_max = 0, 1, 0, 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax.axis("off")
    ax.set_title("Decision Boundary")
    #ax.scatter(x[0, :], x[1, :], c=y, cmap=plt.cm.RdBu)

    layer_dims = [2, 4, 12, 4, 1]
    learning_rate = 0.0075
    iterations = 12000

    nn_model = NeuralNet()

    Z_list = iter_Z(nn_model, x, y, xx, yy)

    time.sleep(5)

    for Z in Z_list:
        ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        ax.scatter(x[0, :], x[1, :], c=y, cmap=plt.cm.RdBu)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)


def stop_train():
    pass

clear_btn = CTkButton(master=frame1, text="Clear", command=clear_canvas)
clear_btn.pack(pady=12, padx=10)


# PARAMETERS

# Number of layers
layers_label = CTkLabel(text="1 hidden layer(s)", master=frame1)
layers_label.pack(pady=12, padx=10)
def set_layers(value):
    layers_label.configure(text=f"{round(value)} hidden layer(s)")

layers_slider = CTkSlider(master=frame1, command=set_layers, from_=1, to=6)
layers_slider.set(1)
layers_slider.pack(pady=12, padx=10)




# Frame 2
train_btn = CTkButton(master=frame2, text="Train", command=train)
train_btn.pack(pady=12, padx=10)

stop_btn = CTkButton(master=frame2, text="Stop", command=stop_train)
stop_btn.pack(pady=12, padx=10)







root.mainloop()