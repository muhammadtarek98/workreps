import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df_trash = pd.read_csv(filepath_or_buffer=r"D:\graval detection project\mareim runs\train7\results.csv")
print(df_trash.keys())
df_stone = pd.read_csv(filepath_or_buffer=r"D:\graval detection project\mareim runs\segment\train_UW_1080\results.csv")


def plot_loss(epochs, loss, loss_type):
    plt.plot(epochs, loss)
    plt.xlabel("epoch")
    plt.ylabel(loss_type)
    plt.title(loss_type)
    plt.grid()
    plt.show()

fig1 = plt.figure()

plot_loss(epochs=df_trash['                  epoch'], loss=df_trash['         train/seg_loss'],
          loss_type="trash material training seg loss")

plot_loss(epochs=df_stone['                  epoch'], loss=df_stone['         train/seg_loss'],
          loss_type="stone training seg loss")


plot_loss(epochs=df_trash['                  epoch'], loss=df_trash['           val/seg_loss'],
          loss_type="trash validation seg loss")
plot_loss(epochs=df_stone['                  epoch'], loss=df_stone['           val/seg_loss'],
          loss_type="stone validation seg loss")

plot_loss(epochs=df_trash['                  epoch'], loss=df_trash['         train/box_loss'],
          loss_type="training box loss")
plot_loss(epochs=df_stone['                  epoch'], loss=df_stone['         train/box_loss'],
          loss_type="training box loss")

plot_loss(epochs=df_trash['                  epoch'], loss=df_trash['           val/box_loss'],
          loss_type="validation bos loss")
plot_loss(epochs=df_stone['                  epoch'], loss=df_stone['           val/box_loss'],
          loss_type="validation bos loss")
