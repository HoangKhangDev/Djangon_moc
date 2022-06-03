# LogisticRegression
from importlib.resources import path
import math
import os
import csv
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
import pylab as pl
import shutil
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import base64


from datetime import datetime
path_stock = os.path.dirname(os.path.abspath("hello"))+os.sep


def Train_batch(batch, path_stock):

    arr = []
    with open(path_stock+"/data.csv") as f:
        arr = f.readlines()
        arr = np.array(arr)
        random.shuffle(arr)
        arr = np.array_split(arr, batch)
    if os.path.exists(path_stock+"/tmp"):
        shutil.rmtree(path_stock+"/tmp")
    os.makedirs(path_stock+"/tmp")
    for i in range(0, len(arr)):
        with open(
                path_stock+f"/tmp/{i+1}.csv", "w") as f:
            f.writelines(arr[i])
    return batch
  # chia thành 10
# n = 20
# bước nhảy k=
# k = 3

def Select_Model():
        Model = GaussianNB()


def MODEL(path_open, path_save, n=20, k=3, Name_Model="GaussianNB"):
    # read file csv
    data_train = pd.read_csv(path_open).values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_save + \
        f'{Name_Model}/{str_date}'+os.path.sep
    if(os.path.exists(path_save + f'{Name_Model}/{str_date}') == False):
        os.makedirs(path_save +
                    f'{Name_Model}/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    arr_nu=[]
    arr_gamma=[]
    x_ve = []
    if(os.path.exists(path_save+f'bieudo') == False):
        os.makedirs(path_save+f'bieudo')
    fs = open(path_save+f'bieudo/{Name_Model}.csv', "w")
    fs.write("")
    fs = open(path_save+f'bieudo/{Name_Model}.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        arr_nu.append(trainx)
        arr_gamma.append(trainy)
        fs.write(f"{i} {accuracy}\n")

    # plt.plot(x_ve, arr)
    # plt.title(Name_Model + str_date)
    # os.makedirs(path_save_parent+'images')
    # plt.savefig(path_save_parent+f"images/{Name_Model}.png")
    # fs = open(path_save_parent+f"images/{Name_Model}.png","rb")
    # encode_base64_image = base64.b64encode(fs.read())
    return arr_nu, arr_gamma, arr, x_ve


def GaussianNB_func(name_file_open="data.csv", n=20, k=3):
    Model = GaussianNB()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/GaussianNB/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/GaussianNB/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/GaussianNB/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/GaussianNB.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("GaussianNB" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/GaussianNB.png")
    plt.show()


def DecisionTreeClassifier_func(name_file_open="data.csv", n=20, k=3):
    Model = DecisionTreeClassifier()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/DecisionTreeClassifier/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/DecisionTreeClassifier/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/DecisionTreeClassifier/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/DecisionTreeClassifier.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("DecisionTreeClassifier" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/DecisionTreeClassifier.png")
    plt.show()


def KNeighborsClassifier_func(name_file_open="data.csv", n=20, k=3):
    Model = KNeighborsClassifier(n_neighbors=8)
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/KNeighborsClassifier/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/KNeighborsClassifier/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/KNeighborsClassifier/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/KNeighborsClassifier.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("KNeighborsClassifier" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/KNeighborsClassifier.png")
    plt.show()


def BernoulliNB_func(name_file_open="data.csv", n=20, k=3):
    Model = BernoulliNB()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/BernoulliNB/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/Test/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/BernoulliNB/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/BernoulliNB.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("BernoulliNB" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/BernoulliNB.png")
    plt.show()


def ExtraTreeClassifier_func(name_file_open="data.csv", n=20, k=3):
    Model = ExtraTreeClassifier()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/ExtraTreeClassifier/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/ExtraTreeClassifier/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/ExtraTreeClassifier/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/ExtraTreeClassifier.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("ExtraTreeClassifier" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/ExtraTreeClassifier.png")
    plt.show()


def BaggingClassifier_func(name_file_open="data.csv", n=20, k=3):
    Model = BaggingClassifier()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/BaggingClassifier/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/BaggingClassifier/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/BaggingClassifier/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/BaggingClassifier.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("BaggingClassifier" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/BaggingClassifier.png")
    plt.show()


def AdaBoostClassifier_func(name_file_open="data.csv", n=20, k=3):
    Model = AdaBoostClassifier()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/AdaBoostClassifier/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/AdaBoostClassifier/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/AdaBoostClassifier/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/AdaBoostClassifier.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("AdaBoostClassifier" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/AdaBoostClassifier.png")
    plt.show()


def MLPClassifier_func(name_file_open="data.csv", n=20, k=3):
    Model = MLPClassifier()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/MLPClassifier/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/MLPClassifier/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/MLPClassifier/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/MLPClassifier.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("MLPClassifier" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/MLPClassifier.png")
    plt.show()


def LinearDiscriminantAnalysis_func(name_file_open="data.csv", n=20, k=3):
    Model = LinearDiscriminantAnalysis()
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/LinearDiscriminantAnalysis/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/LinearDiscriminantAnalysis/{str_date}') == False):
        os.makedirs(path_stock +
                    f'algorithm/LinearDiscriminantAnalysis/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/Test.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("LinearDiscriminantAnalysis" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/LinearDiscriminantAnalysis.png")
    plt.show()


def RandomForestClassifier_func(name_file_open="data.csv", n=20, k=3):
    Model = RandomForestClassifier(max_depth=2)
    data_train = pd.read_csv(path_stock+f"data/{name_file_open}").values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_stock + \
        f'algorithm/RandomForestClassifier/{str_date}'+os.path.sep
    if(os.path.exists(path_stock + f'algorithm/RandomForestClassifier/{str_date}') == False):
        os.makedirs(
            path_stock + f'algorithm/RandomForestClassifier/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    x_ve = []
    if(os.path.exists(path_stock+f'bieudo') == False):
        os.makedirs(path_stock+f'bieudo')
    fs = open(path_stock+f'bieudo/RandomForestClassifier.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i)
        fs.write(f"{i} {accuracy}\n")

    plt.plot(x_ve, arr)
    plt.title("RandomForestClassifier" + str_date)
    os.makedirs(path_save_parent+'images')
    plt.savefig(path_save_parent+"images/Test.png")
    plt.show()


# os.shutil.rmtree(path_stock+"tmp")
# os.shutil.rmtree(path_stock+"bieudo")
# os.shutil.rmtree(path_stock+"algorithm")


# Train_batch(10)
# GaussianNB_func()
# DecisionTreeClassifier_func()
# KNeighborsClassifier_func()
# BernoulliNB_func()
# ExtraTreeClassifier_func()
# BaggingClassifier_func()
# AdaBoostClassifier_func()
# MLPClassifier_func()
# LinearDiscriminantAnalysis_func()
# RandomForestClassifier_func()
# MODEL(path_stock+'/data/data.csv', path_stock, 10, 3, "GaussianNB")
