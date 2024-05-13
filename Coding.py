from xml.parsers.expat import model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog 

import os

from PyQt5.QtCore import QStringListModel

from matplotlib import image
from sklearn.model_selection import train_test_split
from shutil import copyfile

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


from keras import backend as B
from keras.applications import VGG16
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.applications import Xception
from tensorflow.keras.applications import ResNet50

from PyQt5 import QtCore, QtGui, QtWidgets

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from shutil import copyfile


from PyQt5 import QtCore, QtGui, QtWidgets

from sklearn.model_selection import StratifiedKFold




class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(995, 1016)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 39, 70, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 64, 64, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(140, 35, 150, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(140, 85, 150, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(650, 5, 301, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        self.label_4.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(40, 333, 43, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(40, 358, 84, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(141, 326, 150, 22))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_7.setGeometry(QtCore.QRect(141, 354, 150, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_7.setFont(font)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 575, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 695, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(145, 308, 58, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(213, 308, 71, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setEnabled(True)
        self.label_9.setGeometry(QtCore.QRect(680, 255, 250, 250))
        self.label_9.setStyleSheet("background-color: gray;")
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_9.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_9.setFont(font)
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(40, 190, 50, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(40, 140, 79, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(40, 165, 107, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(40, 415, 81, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_8.setGeometry(QtCore.QRect(141, 186, 150, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_8.setFont(font)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_9.setGeometry(QtCore.QRect(141, 136, 150, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_9.setFont(font)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_10.setGeometry(QtCore.QRect(141, 161, 150, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_10.setFont(font)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_11 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_11.setGeometry(QtCore.QRect(141, 415, 150, 21))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setEnabled(True)
        self.textEdit.setGeometry(QtCore.QRect(350, 35, 251, 201))
        self.textEdit.setObjectName("textEdit")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(350, 5, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(40, 755, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(40, 815, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(40, 308, 54, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(390, 615, 171, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(350, 665, 250, 250))
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.label_18.setStyleSheet("background-color: gray;")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(40, 5, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(350, 955, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(350, 265, 261, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setEnabled(True)
        self.textEdit_2.setGeometry(QtCore.QRect(350, 305, 251, 201))
        self.textEdit_2.setObjectName("textEdit_2")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(700, 545, 171, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(660, 885, 300, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        self.label_27.setGeometry(QtCore.QRect(660, 915, 300, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.centralwidget)
        self.label_28.setGeometry(QtCore.QRect(660, 945, 300, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.label_29 = QtWidgets.QLabel(self.centralwidget)
        self.label_29.setGeometry(QtCore.QRect(680, 605, 250, 250))
        self.label_29.setText("")
        self.label_29.setObjectName("label_29")
        self.label_29.setStyleSheet("background-color: gray;")
        self.label_37 = QtWidgets.QLabel(self.centralwidget)
        self.label_37.setGeometry(QtCore.QRect(660, 975, 300, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(40, 635, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setAutoFillBackground(False)
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_30 = QtWidgets.QLabel(self.centralwidget)
        self.label_30.setGeometry(QtCore.QRect(40, 265, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.centralwidget)
        self.label_31.setGeometry(QtCore.QRect(350, 575, 261, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.centralwidget)
        self.label_32.setGeometry(QtCore.QRect(650, 505, 261, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(650, 215, 311, 25))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_21 = QtWidgets.QLabel(self.splitter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.radioButton_3 = QtWidgets.QRadioButton(self.splitter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.splitter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_5 = QtWidgets.QRadioButton(self.splitter)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.radioButton_5.setFont(font)
        self.radioButton_5.setObjectName("radioButton_5")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(650, 45, 301, 151))
        self.listView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.listView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.listView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.listView.setAutoScroll(True)
        self.listView.setObjectName("listView")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 89, 102, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(140, 60, 150, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        mainWindow.setCentralWidget(self.centralwidget)
        self.actionG_z_At = QtWidgets.QAction(mainWindow)
        self.actionG_z_At.setObjectName("actionG_z_At")


        self.radioButton_3.clicked.connect(self.update_list_view)
        self.radioButton_4.clicked.connect(self.update_list_view) 
        self.radioButton_5.clicked.connect(self.update_list_view) 
        self.listView.clicked.connect(self.display_selected_image) 
      
        self.pushButton.clicked.connect(self.cnn_model_hold_out) 
        self.pushButton_7.clicked.connect(self.cnn_model_k_fold) 
        self.pushButton_2.clicked.connect(lambda: self.model_egit(VGG16))
        self.pushButton_3.clicked.connect(lambda: self.model_egit(VGG19))
        self.pushButton_4.clicked.connect(lambda: self.model_egit(ResNet50))
        self.pushButton_5.clicked.connect(self.choose_image)
        self.pushButton_6.clicked.connect(self.choose_image2)   

        




        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        mainWindow.setTabOrder(self.listView, self.radioButton)
        mainWindow.setTabOrder(self.radioButton, self.lineEdit_6)
        mainWindow.setTabOrder(self.lineEdit_6, self.lineEdit_7)
        mainWindow.setTabOrder(self.lineEdit_7, self.pushButton)
        mainWindow.setTabOrder(self.pushButton, self.pushButton_2)
        mainWindow.setTabOrder(self.pushButton_2, self.lineEdit_8)
        mainWindow.setTabOrder(self.lineEdit_8, self.lineEdit_2)
        mainWindow.setTabOrder(self.lineEdit_2, self.lineEdit_9)
        mainWindow.setTabOrder(self.lineEdit_9, self.pushButton_3)
        mainWindow.setTabOrder(self.pushButton_3, self.lineEdit_10)
        mainWindow.setTabOrder(self.lineEdit_10, self.lineEdit_11)
        mainWindow.setTabOrder(self.lineEdit_11, self.lineEdit)
        mainWindow.setTabOrder(self.lineEdit, self.radioButton_2)
        mainWindow.setTabOrder(self.radioButton_2, self.lineEdit_3)
        mainWindow.setTabOrder(self.lineEdit_3, self.textEdit)
        mainWindow.setTabOrder(self.textEdit, self.pushButton_4)
        mainWindow.setTabOrder(self.pushButton_4, self.radioButton_3)
        mainWindow.setTabOrder(self.radioButton_3, self.radioButton_4)
        mainWindow.setTabOrder(self.radioButton_4, self.radioButton_5)
        mainWindow.setTabOrder(self.radioButton_5, self.pushButton_5)
        mainWindow.setTabOrder(self.pushButton_5, self.textEdit_2)
        mainWindow.setTabOrder(self.textEdit_2, self.pushButton_6)
        mainWindow.setTabOrder(self.pushButton_6, self.pushButton_7)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "MainWindow"))
        self.label.setText(_translate("mainWindow", "Train Set :"))
        self.label_2.setText(_translate("mainWindow", "Test Set :"))
        self.label_4.setText(_translate("mainWindow", "  Train ,Test, Validation Resimler "))
        self.label_7.setText(_translate("mainWindow", "Zoom :"))
        self.label_8.setText(_translate("mainWindow", "Shear Range :"))
        self.pushButton.setText(_translate("mainWindow", "CNN hold_out"))
        self.pushButton_2.setText(_translate("mainWindow", "VGG16"))
        self.radioButton.setText(_translate("mainWindow", "Vertical"))
        self.radioButton_2.setText(_translate("mainWindow", "Horizantal"))
        self.label_10.setText(_translate("mainWindow", "Epoch :"))
        self.label_11.setText(_translate("mainWindow", "Batch Size :"))
        self.label_12.setText(_translate("mainWindow", "Early Stopping :"))
        self.label_13.setText(_translate("mainWindow", "Model Adı :"))
        self.label_14.setText(_translate("mainWindow", "           Eğitim Sonuçları"))
        self.pushButton_3.setText(_translate("mainWindow", "VGG19"))
        self.pushButton_4.setText(_translate("mainWindow", "ResNet50"))
        self.label_17.setText(_translate("mainWindow", "Çevirme:"))
        self.pushButton_5.setText(_translate("mainWindow", "Resim Seçiniz"))
        self.label_20.setText(_translate("mainWindow", "Veri Seti Üzerinde İşlemler"))
        self.label_22.setText(_translate("mainWindow", "Tahmin Sonucu: "))
        self.label_23.setText(_translate("mainWindow", "Konfisyon Matrisi - Metrikler "))
        self.pushButton_6.setText(_translate("mainWindow", "Chose Image :"))
        self.label_26.setText(_translate("mainWindow", "Prediction VGG16 :"))
        self.label_27.setText(_translate("mainWindow", "Prediction VGG19 :"))
        self.label_28.setText(_translate("mainWindow", "Prediction ResNet50 :"))
        self.label_37.setText(_translate("mainWindow", "Avarage Prediction :"))
        self.pushButton_7.setText(_translate("mainWindow", "CNN k_fold"))
        self.label_30.setText(_translate("mainWindow", "      Veri Çoklama İşlemi"))
        self.label_31.setText(_translate("mainWindow", "           CNN TAHMİN"))
        self.label_32.setText(_translate("mainWindow", "      ORTALAMA TAHMİN"))
        self.label_21.setText(_translate("mainWindow", "Seçiniz:"))
        self.radioButton_3.setText(_translate("mainWindow", "Train"))
        self.radioButton_4.setText(_translate("mainWindow", "Test"))
        self.radioButton_5.setText(_translate("mainWindow", "Validation"))
        self.label_3.setText(_translate("mainWindow", "Validation Set :"))
        self.actionG_z_At.setText(_translate("mainWindow", "Göz at"))


    def update_list_view(self): 
        selected_folder = ""

        if self.radioButton_3.isChecked(): 
            selected_folder = "dataset_split/train/"
        elif self.radioButton_4.isChecked():
            selected_folder = "dataset_split/test/"
        elif self.radioButton_5.isChecked():  
            selected_folder = "dataset_split/validation/"

        self.update_list_view_contents(selected_folder)

    def update_list_view_contents(self, selected_folder): 
        model = QStringListModel()
        file_list = []

        for root, dirs, files in os.walk(selected_folder):
            for file in files:
                file_list.append(os.path.join(root, file))

        model.setStringList(file_list)
        self.listView.setModel(model)

    def display_selected_image(self, index):
        selected_item = self.listView.model().index(index.row(), 0)
        file_path = self.listView.model().data(selected_item, QtCore.Qt.DisplayRole)

        pixmap = QtGui.QPixmap(file_path)
        pixmap = pixmap.scaledToWidth(250)  
        self.label_9.setPixmap(pixmap)

    def update_text_edit(self,history):
        self.textEdit.clear()
        self.textEdit.append(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        self.textEdit.append(f"Training Loss: {history.history['loss'][-1]:.4f}")
        self.textEdit.append(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        self.textEdit.append(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")

    def choose_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self.centralwidget, "Choose Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)

        if file_path:
            pixmap = QtGui.QPixmap(file_path)
            pixmap = pixmap.scaledToWidth(250)  
            self.label_18.setPixmap(pixmap)
        self.predict_image(file_path)
        
    
    def choose_image2(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self.centralwidget, "Choose Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)

        if file_path:
            pixmap = QtGui.QPixmap(file_path)
            pixmap = pixmap.scaledToWidth(250)  
            self.label_29.setPixmap(pixmap)
        self.predict_image2(file_path)

        

    def cnn_model_hold_out(self):
        train_size_text = int(self.lineEdit.text())
        test_size_text = int(self.lineEdit_2.text())
        validation_size_text = int(self.lineEdit_3.text())
        epochs_text = int(self.lineEdit_8.text())
        batch_size_text = int(self.lineEdit_9.text())
        early_stopping_patience_text = int(self.lineEdit_10.text())
        model_name_text = str(self.lineEdit_11.text())
        zoom_text = float(self.lineEdit_6.text())
        shear_text = float(self.lineEdit_7.text())

        img_width, img_height = 224, 224

        original_dataset_dir = "dataset"

        base_dir = "dataset_split"
        os.makedirs(base_dir, exist_ok=True)

        class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']

        train_dir = os.path.join(base_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)

        validation_dir = os.path.join(base_dir, 'validation')
        os.makedirs(validation_dir, exist_ok=True)

        test_dir = os.path.join(base_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)

        for class_name in class_names:
            class_dir = os.path.join(original_dataset_dir, class_name)
    
            train_class_dir, test_class_dir = train_test_split(os.listdir(class_dir), test_size=test_size_text / 100, random_state=42)
            validation_class_dir, train_class_dir = train_test_split(train_class_dir, test_size=(10-(validation_size_text / 10))/10, random_state=42)
            
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
            for file_name in train_class_dir:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(train_dir, class_name, file_name)
                copyfile(src, dst)
    
            for file_name in validation_class_dir:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(validation_dir, class_name, file_name)
                copyfile(src, dst)
    
            for file_name in test_class_dir:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(test_dir, class_name, file_name)
                copyfile(src, dst)

        if B.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)


        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (5, 5)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4))  
        model.add(Activation('softmax')) 
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        
        horizontal_flip = self.radioButton_2.isChecked()
        vertical_flip = self.radioButton.isChecked()

        if horizontal_flip or vertical_flip:
            train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=float(shear_text),
            zoom_range=float(zoom_text),
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip
        )


        test_data_gen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_data_gen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size_text,
            class_mode='categorical')

        validation_generator = test_data_gen.flow_from_directory(
            validation_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size_text,
            class_mode='categorical')

        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience_text,mode='max',restore_best_weights=True)  
        model_checkpoint = ModelCheckpoint(model_name_text + '.h5',monitor ='val_accuracy', save_best_only=True)

        history = model.fit(
            train_generator,
            steps_per_epoch= len(train_class_dir) // batch_size_text,
            epochs=epochs_text,
            validation_data=validation_generator,
            validation_steps= len(validation_class_dir) // batch_size_text,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        self.update_text_edit(history)

        test_dir = "dataset_split/test/"
        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_data_gen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        predictions = model.predict(test_generator)
        true_labels = test_generator.classes

        predicted_classes = np.argmax(predictions, axis=1)
        cm = confusion_matrix(true_labels, predicted_classes)
        report = classification_report(true_labels, predicted_classes, zero_division=1)


        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['cataract','diabetic_retinopathy','glaucoma','normal'], yticklabels=['cataract','diabetic_retinopathy','glaucoma','normal'])
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')
        plt.show()

        
        print(report)

        
        self.textEdit_2.clear()
        self.textEdit_2.append(f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{report}")
        

    def cnn_model_k_fold(self):
        train_size_text = int(self.lineEdit.text())
        test_size_text = int(self.lineEdit_2.text())
        validation_size_text = int(self.lineEdit_3.text())
        epochs_text = int(self.lineEdit_8.text())
        batch_size_text = int(self.lineEdit_9.text())
        early_stopping_patience_text = int(self.lineEdit_10.text())
        model_name_text = str(self.lineEdit_11.text())
        zoom_text = float(self.lineEdit_6.text())
        shear_text = float(self.lineEdit_7.text())

        img_width, img_height = 224, 224
        original_dataset_dir = "dataset"
        base_dir = "dataset_split"
        os.makedirs(base_dir, exist_ok=True)
        class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        
        for fold, (train_index, test_index) in enumerate(kfold.split(np.zeros(len(class_names)), np.zeros(len(class_names)))):
            train_dir = os.path.join(base_dir, f'train_fold_{fold}')
            os.makedirs(train_dir, exist_ok=True)
            validation_dir = os.path.join(base_dir, f'validation_fold_{fold}')
            os.makedirs(validation_dir, exist_ok=True)
            test_dir = os.path.join(base_dir, f'test_fold_{fold}')
            os.makedirs(test_dir, exist_ok=True)

            for class_name in class_names:
                class_dir = os.path.join(original_dataset_dir, class_name)

                train_class_dir, test_class_dir = train_test_split(os.listdir(class_dir), test_size=test_size_text / 100,
                                                                    random_state=42)
                validation_class_dir, train_class_dir = train_test_split(train_class_dir,
                                                                         test_size=(10 - (validation_size_text / 10)) / 10,
                                                                         random_state=42)

                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

                for file_name in train_class_dir:
                    src = os.path.join(class_dir, file_name)
                    dst = os.path.join(train_dir, class_name, file_name)
                    copyfile(src, dst)

                for file_name in validation_class_dir:
                    src = os.path.join(class_dir, file_name)
                    dst = os.path.join(validation_dir, class_name, file_name)
                    copyfile(src, dst)

                for file_name in test_class_dir:
                    src = os.path.join(class_dir, file_name)
                    dst = os.path.join(test_dir, class_name, file_name)
                    copyfile(src, dst)

            if B.image_data_format() == 'channels_first':
                input_shape = (3, img_width, img_height)
            else:
                input_shape = (img_width, img_height, 3)

            model = Sequential()
            model.add(Conv2D(32, (3, 3), input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (5, 5)))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(64))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(4))  
            model.add(Activation('softmax')) 
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            
            horizontal_flip = self.radioButton_2.isChecked()
            vertical_flip = self.radioButton.isChecked()

            if horizontal_flip or vertical_flip:
                train_data_gen = ImageDataGenerator(
                    rescale=1. / 255,
                    shear_range=float(shear_text),
                    zoom_range=float(zoom_text),
                    horizontal_flip=horizontal_flip,
                    vertical_flip=vertical_flip
                )

            test_data_gen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_data_gen.flow_from_directory(
                train_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size_text,
                class_mode='categorical')

            validation_generator = test_data_gen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size_text,
                class_mode='categorical')
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience_text,mode='max',restore_best_weights=True)  
            model_checkpoint = ModelCheckpoint(model_name_text + '.h5',monitor ='val_accuracy', save_best_only=True)

            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_class_dir) // batch_size_text,
                epochs=epochs_text,
                validation_data=validation_generator,
                validation_steps=len(validation_class_dir) // batch_size_text,
                callbacks=[early_stopping, model_checkpoint]
            )

            self.update_text_edit(history)

            test_data_gen = ImageDataGenerator(rescale=1. / 255)
            test_generator = test_data_gen.flow_from_directory(
                test_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )

            predictions = model.predict_generator(test_generator)
            true_labels = test_generator.classes

            predicted_classes = np.argmax(predictions, axis=1)
            cm = confusion_matrix(true_labels, predicted_classes)
            report = classification_report(true_labels, predicted_classes, zero_division=1)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'],
                        yticklabels=['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'])
            plt.xlabel('Tahmin Edilen Sınıf')
            plt.ylabel('Gerçek Sınıf')
            plt.title('Karmaşıklık Matrisi')
            plt.show()

            print(report)

            self.textEdit_2.clear()
            self.textEdit_2.append(f"Karmaşıklık Matrisi:\n{cm}\n\nSınıflandırma Raporu:\n{report}")
        
    
    
    def model_egit(self,pre_trained):
         
        train_size_text = int(self.lineEdit.text())
        test_size_text = int(self.lineEdit_2.text())
        validation_size_text = int(self.lineEdit_3.text())
        epochs_text = int(self.lineEdit_8.text())
        batch_size_text = int(self.lineEdit_9.text())
        early_stopping_patience_text = int(self.lineEdit_10.text())
        model_name_text = str(self.lineEdit_11.text())
        zoom_text = float(self.lineEdit_6.text())
        shear_text = float(self.lineEdit_7.text())

        img_width, img_height = 224, 224

        original_dataset_dir = "dataset"

        base_dir = "dataset_split"
        os.makedirs(base_dir, exist_ok=True)

        class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']

        train_dir = os.path.join(base_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)

        validation_dir = os.path.join(base_dir, 'validation')
        os.makedirs(validation_dir, exist_ok=True)

        test_dir = os.path.join(base_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)

        for class_name in class_names:
            class_dir = os.path.join(original_dataset_dir, class_name)

            train_class_dir, test_class_dir = train_test_split(os.listdir(class_dir), test_size=test_size_text / 100, random_state=42)
            validation_class_dir, train_class_dir = train_test_split(train_class_dir, test_size=(10-(validation_size_text / 10))/10, random_state=42)

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            for file_name in train_class_dir:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(train_dir, class_name, file_name)
                copyfile(src, dst)

            for file_name in validation_class_dir:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(validation_dir, class_name, file_name)
                copyfile(src, dst)

            for file_name in test_class_dir:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(test_dir, class_name, file_name)
                copyfile(src, dst)

        pre_trained_model = pre_trained(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

        for layer in pre_trained_model.layers:
            layer.trainable = False

        model = Sequential()
        model.add(pre_trained_model)
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        horizontal_flip = self.radioButton_2.isChecked()
        vertical_flip = self.radioButton.isChecked()

        
        if horizontal_flip or vertical_flip:
            train_data_gen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=float(shear_text),
                zoom_range=float(zoom_text),
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip
            )

        test_data_gen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_data_gen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size_text,
            class_mode='categorical')

        validation_generator = test_data_gen.flow_from_directory(
            validation_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size_text,
            class_mode='categorical')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience_text,mode='max',restore_best_weights=True)  
        model_checkpoint = ModelCheckpoint(model_name_text + '.h5',monitor ='val_accuracy', save_best_only=True)

        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_class_dir) // batch_size_text,
            epochs=epochs_text,
            validation_data=validation_generator,
            validation_steps=len(validation_class_dir) // batch_size_text,
            callbacks=[early_stopping, model_checkpoint]
        )

        self.update_text_edit(history)

        test_dir = "dataset_split/test/"
        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_data_gen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        
        predictions = model.predict(test_generator)
        true_labels = test_generator.classes

        predicted_classes = np.argmax(predictions, axis=1)
        cm = confusion_matrix(true_labels, predicted_classes)
        report = classification_report(true_labels, predicted_classes, zero_division=1)


        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['cataract','diabetic_retinopathy','glaucoma','normal'], yticklabels=['cataract','diabetic_retinopathy','glaucoma','normal'])
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')
        plt.show()

        
        print(report)

        
        self.textEdit_2.clear()
        self.textEdit_2.append(f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{report}")
        
        
    def predict_image2(self, file_path):

        trained_model = load_model('VGG16_model.h5')
 
        img = load_img(file_path, target_size=(224, 224))
        
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = trained_model.predict(img_array)

        predicted_class_index = np.argmax(predictions[0])
        class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']
        predicted_class = class_names[predicted_class_index]
        
        max_index = np.argmax(predictions)
        max_value = predictions[0, max_index]
        decimal_part = str(max_value).split(".")[1][:2]
        self.label_26.setText(f"VGG16 Tahmini: {predicted_class} % {decimal_part}")

        test_dir = "dataset_split/test/"
        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_data_gen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        
        trained_model = load_model('VGG19_model.h5')
 
        img = load_img(file_path, target_size=(224, 224))
        
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = trained_model.predict(img_array)

        predicted_class_index = np.argmax(predictions[0])
        class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']
        predicted_class2 = class_names[predicted_class_index]
        
        max_index = np.argmax(predictions)
        max_value = predictions[0, max_index]
        decimal_part2 = str(max_value).split(".")[1][:2]
        self.label_27.setText(f"VGG19 Tahmini: {predicted_class2} % {decimal_part2}")

        test_dir = "dataset_split/test/"
        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_data_gen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        #

        trained_model = load_model('RESNET50_model.h5')
 
        img = load_img(file_path, target_size=(224, 224))
        
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = trained_model.predict(img_array)

        predicted_class_index = np.argmax(predictions[0])
        class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']
        predicted_class3 = class_names[predicted_class_index]
        
        max_index = np.argmax(predictions)
        max_value = predictions[0, max_index]
        decimal_part3 = str(max_value).split(".")[1][:2]
        self.label_28.setText(f"RESNET50 Tahmini: {predicted_class3} % {decimal_part3}")

        test_dir = "dataset_split/test/"
        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_data_gen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )


        tahminler = [predicted_class, predicted_class2, predicted_class3]
        
        tsonuc = max(set(tahminler), key = tahminler.count)
        
        self.label_37.setText(f"Ortalama Sonucu:{tsonuc}")
        


    def predict_image(self, file_path):

        trained_model = load_model('CNN_model.h5')  

         
        img = load_img(file_path, target_size=(224, 224))
        
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = trained_model.predict(img_array)

        predicted_class_index = np.argmax(predictions[0])
        class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']
        predicted_class = class_names[predicted_class_index]
        
        max_index = np.argmax(predictions)
        max_value = predictions[0, max_index]
        decimal_part4 = str(max_value).split(".")[1][:2]
        self.label_22.setText(f"Tahmin Sonucu: {predicted_class} % {decimal_part4}")

        test_dir = "dataset_split/test/"
        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_data_gen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

               

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())