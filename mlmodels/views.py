from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
import json
import datetime
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_protect
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score
from mlTester.settings import BASE_DIR

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import codecs
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense

from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication


def mltester(request):
    return render(request, 'index.html')


@api_view(['GET', 'POST'])
def linear_regression(request):
    context = {'error': "Error...."}
    if request.method == 'POST':
        checked_box, train, test, lb_col = read_posted_request(request)

        norm, intercpt = find_norm_intercpt(checked_box)

        data = codecs.EncodedFile(train.open(), "utf-8")
        data = pd.read_csv(data)

        data.drop([i for i in data.columns if "nnamed" in i], axis=1, inplace=True)
        data = label_encoding(data)

        X, y = data[[i for i in data.columns if i != lb_col]], data[[lb_col]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        reg = LinearRegression(fit_intercept=intercpt, normalize=norm).fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Plotting parameters
        fig, axes = plt.subplots(figsize=(5, 5))

        title = "Learning Curves (Linear Regression)"
        cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)

        estimator = LinearRegression()
        plot_learning_curve(estimator, title, X, y, axes, ylim=(0.1, 1.01),
                            cv=cv, n_jobs=4)

        plt.savefig(BASE_DIR / 'static/images/lin_reg_image.jpg')

        context['error'] = {"Mean absolute error": metrics.mean_absolute_error(y_test[lb_col], y_pred),
                            "Mean squares error": metrics.mean_squared_error(y_test[lb_col], y_pred)}

        return Response(context)
    return render(request, "linear_regression.html", context)


@api_view(['GET', 'POST'])
def logistic_regression(request):
    context = {'error': "Error...."}
    if request.method == 'POST':

        checked_box, train, test, lb_col = read_posted_request(request)
        print("Data extracted from request obj!")
        norm, intercpt = find_norm_intercpt(checked_box)

        data = codecs.EncodedFile(train.open(), "utf-8")
        data = pd.read_csv(data)

        label_mapping = get_label_mapping(data[[lb_col]])
        data.drop([i for i in data.columns if "nnamed" in i], axis=1, inplace=True)
        data = label_encoding(data)
        print("label encoding Done!")
        X, y = data[[i for i in data.columns if i != lb_col]], data[[lb_col]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        y_train, y_test, y = np.ravel(y_train), np.ravel(y_test), np.ravel(y)
        print("LoR Begins....")
        reg = LogisticRegression(max_iter=200, fit_intercept=intercpt).fit(X_train, y_train)
        print("LoR Done.")
        y_pred = reg.predict(X_test)
        if X.shape[0] < 4000:
            print("Learning curve plot begins....")
            # Plotting parameters
            fig, axes = plt.subplots(figsize=(5, 5))

            title = "Learning Curves (Logistic Regression)"
            cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

            estimator = LogisticRegression(max_iter=200)
            plot_learning_curve(estimator, title, X, y, axes, ylim=(0.1, 1.01),
                                cv=cv, n_jobs=4)
            plt.savefig(BASE_DIR / 'static/images/log_reg_image.jpg')
            print("Learning curve plotted!")

            print("Visualization begin....")
            visualize_dataset(X, y, label_mapping)
            print("Visualization done.")
        else:
            file1 = BASE_DIR / 'static/images/log_reg_image.jpg'
            file2 = BASE_DIR / 'static/images/log_reg_vis_image.jpg'
            if os.path.exists(file1):
                os.remove(file1)
            else:
                print("No such image....! ", file1)
            if os.path.exists(file2):
                os.remove(file2)
            else:
                print("No such image....! ", file2)

        context['error'] = {"Mean absolute error": metrics.mean_absolute_error(y_test, y_pred),
                            "Mean squares error": metrics.mean_squared_error(y_test, y_pred),
                            "F1-score": f1_score(y_test, y_pred, average='micro')}
        return Response(context)

    return render(request, "logistic_regression.html", context)


def svm(request):
    return HttpResponse("<h1>svm</h1>")


def decision_tree(request):
    return HttpResponse("<h1>decision_tree</h1>")


@api_view(['GET', 'POST'])
def custom_neural_network(request):
    if request.method == 'POST':

        # Request read.....................
        layers = request.data['layers']
        data = request.FILES['file']
        lb_col = request.POST['gt']
        loss = request.POST['loss']
        # Request read end.................

        # Variable Definition..............
        layers = json.loads(layers)
        print("Total layers:", len(layers))

        data = codecs.EncodedFile(data.open(), "utf-8")
        data = pd.read_csv(data)

        print("Total layers:", len(layers))

        neuron_list = []
        for layer in layers:
            print(layer.keys())
            neuron_list.append([len(layer['neuron_arr']), (layer["trait"] if layer["trait"] != "" else "0")])

        activation_dict = {"0": "sigmoid",
                           "1": "relu",
                           "2": "softmax",
                           "3": "softplus",
                           "4": "softsign",
                           "5": "dropout"}

        loss_dict = {"0": "binary_crossentropy",
                     "1": "categorical_crossentropy",
                     "2": "sparse_categorical_crossentropy",
                     "3": "mean_absolute_error",
                     "4": "cosine_similarity"}
        # Variable definition end................

        # Some data preprocessing................
        label_mapping = get_label_mapping(data[[lb_col]])
        data.drop([i for i in data.columns if "nnamed" in i], axis=1, inplace=True)
        data = label_encoding(data)
        print("label encoding Done!")

        X, y = data[[i for i in data.columns if i != lb_col]], data[[lb_col]]
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y)
        print("One Hot Done!")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Data preprocessing end.................

        # define the keras model.................
        feature_no = len(X_train.columns)
        model = Sequential()

        for layer_attri in neuron_list:
            print(layer_attri)
            model.add(Dense(layer_attri[0], input_dim=feature_no, activation=activation_dict[layer_attri[1]]))
            feature_no = None
        last_layer_cnt = len(data[lb_col].unique().tolist())
        print("Class count:", last_layer_cnt)
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)
        # Keras model defined...................

        # Model accuracy calculation............
        model.compile(loss=loss_dict[loss], optimizer='adam', metrics=['accuracy'])
        print(X_train.shape, y_train.shape, "train, test shapes")
        model.fit(X_train, y_train, epochs=100, batch_size=10)
        _, accuracy_train = model.evaluate(X_train, y_train)
        _, accuracy_test = model.evaluate(X_test, y_test)
        print('Accuracy train: %.2f' % (accuracy_train * 100))
        print('Accuracy test: %.2f' % (accuracy_test * 100))
        return Response({'summary': short_model_summary, 'train_accuracy': accuracy_train, 'test_accuracy': accuracy_test})
    return render(request, "custom_neural_network.html")


# Utility
def get_label_mapping(y):
    x = LabelEncoder()
    x.fit_transform(y)
    d = dict(zip(x.transform(x.classes_).tolist(), x.classes_.tolist()))
    return d


def visualize_dataset(X, y, d):
    y = y.reshape(len(y), 1)
    tsn = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_tsn = tsn.fit_transform(X)
    X_tsn_y = np.hstack((X_tsn, y))
    DF = pd.DataFrame(data=X_tsn_y, columns=['TSNE_dimn_1', 'TSNE_dimn_2', 'label'])
    DF['label'] = DF['label'].apply(lambda x: d[x])
    sns.FacetGrid(DF, hue='label', height=6, ).map(plt.scatter, 'TSNE_dimn_1', 'TSNE_dimn_2').add_legend()
    plt.savefig(BASE_DIR / 'static/images/log_reg_vis_image.jpg')


def label_encoding(data):
    dt_list = ['objects', 'string']
    for i in data.columns:
        data[i] = LabelEncoder().fit_transform(data[i]) if any(
            [str(data[i].dtypes) for j in dt_list if str(data[i].dtypes) in j]) else data[i]
    return data


def read_posted_request(request):
    checked_box = request.POST.getlist('ch_bx')
    train = request.FILES["train"]
    test = request.FILES.get("test", None)
    lb_col = request.POST['lb_col']
    return checked_box, train, test, lb_col


def find_norm_intercpt(checked_box):
    norm, intercpt = False, False
    for i in checked_box:
        if i == 'normalize': norm = True
        if i == 'intercept': intercpt = True
    print(norm, intercpt)
    return norm, intercpt


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")
