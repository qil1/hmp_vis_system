# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'proj.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QSpinBox, QStatusBar, QWidget)

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        if not mainWindow.objectName():
            mainWindow.setObjectName(u"mainWindow")
        mainWindow.resize(1132, 700)
        icon = QIcon()
        icon.addFile(u"icons/\u7cfb\u7edf_system.svg", QSize(), QIcon.Normal, QIcon.Off)
        mainWindow.setWindowIcon(icon)
        self.centralwidget = QWidget(mainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_5 = QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_3 = QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.widget_video = QVideoWidget(self.groupBox_3)
        self.widget_video.setObjectName(u"widget_video")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_video.sizePolicy().hasHeightForWidth())
        self.widget_video.setSizePolicy(sizePolicy)
        self.widget_video.setMinimumSize(QSize(533, 300))

        self.gridLayout_3.addWidget(self.widget_video, 0, 0, 1, 1)


        self.horizontalLayout_3.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_4 = QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.widget_video_pred = QVideoWidget(self.groupBox_4)
        self.widget_video_pred.setObjectName(u"widget_video_pred")
        sizePolicy.setHeightForWidth(self.widget_video_pred.sizePolicy().hasHeightForWidth())
        self.widget_video_pred.setSizePolicy(sizePolicy)
        self.widget_video_pred.setMinimumSize(QSize(533, 300))

        self.gridLayout_4.addWidget(self.widget_video_pred, 0, 0, 1, 1)


        self.horizontalLayout_3.addWidget(self.groupBox_4)


        self.gridLayout_5.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.bar_slider = QSlider(self.centralwidget)
        self.bar_slider.setObjectName(u"bar_slider")
        self.bar_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout.addWidget(self.bar_slider)

        self.btn_open = QPushButton(self.centralwidget)
        self.btn_open.setObjectName(u"btn_open")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.btn_open.sizePolicy().hasHeightForWidth())
        self.btn_open.setSizePolicy(sizePolicy1)
        icon1 = QIcon()
        icon1.addFile(u"icons/\u6587\u4ef6\u5939-\u5f00_folder-open.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_open.setIcon(icon1)

        self.horizontalLayout.addWidget(self.btn_open)

        self.btn_play = QPushButton(self.centralwidget)
        self.btn_play.setObjectName(u"btn_play")
        sizePolicy1.setHeightForWidth(self.btn_play.sizePolicy().hasHeightForWidth())
        self.btn_play.setSizePolicy(sizePolicy1)
        icon2 = QIcon()
        icon2.addFile(u"icons/\u64ad\u653e_play.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_play.setIcon(icon2)

        self.horizontalLayout.addWidget(self.btn_play)

        self.btn_pause = QPushButton(self.centralwidget)
        self.btn_pause.setObjectName(u"btn_pause")
        sizePolicy1.setHeightForWidth(self.btn_pause.sizePolicy().hasHeightForWidth())
        self.btn_pause.setSizePolicy(sizePolicy1)
        icon3 = QIcon()
        icon3.addFile(u"icons/\u6682\u505c_pause-one.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_pause.setIcon(icon3)

        self.horizontalLayout.addWidget(self.btn_pause)


        self.gridLayout_5.addLayout(self.horizontalLayout, 1, 0, 1, 1)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_choosemodel = QLabel(self.groupBox_2)
        self.label_choosemodel.setObjectName(u"label_choosemodel")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_choosemodel.sizePolicy().hasHeightForWidth())
        self.label_choosemodel.setSizePolicy(sizePolicy2)
        self.label_choosemodel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.label_choosemodel)

        self.comboBox_choosemodel = QComboBox(self.groupBox_2)
        self.comboBox_choosemodel.addItem("")
        self.comboBox_choosemodel.addItem("")
        self.comboBox_choosemodel.addItem("")
        self.comboBox_choosemodel.addItem("")
        self.comboBox_choosemodel.addItem("")
        self.comboBox_choosemodel.setObjectName(u"comboBox_choosemodel")

        self.horizontalLayout_2.addWidget(self.comboBox_choosemodel)

        self.btn_load_weight = QPushButton(self.groupBox_2)
        self.btn_load_weight.setObjectName(u"btn_load_weight")
        sizePolicy1.setHeightForWidth(self.btn_load_weight.sizePolicy().hasHeightForWidth())
        self.btn_load_weight.setSizePolicy(sizePolicy1)
        icon4 = QIcon()
        icon4.addFile(u"icons/\u4ee3\u7801\u6587\u4ef6_file-code.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_load_weight.setIcon(icon4)

        self.horizontalLayout_2.addWidget(self.btn_load_weight)

        self.label_modelstatus = QLabel(self.groupBox_2)
        self.label_modelstatus.setObjectName(u"label_modelstatus")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_modelstatus.sizePolicy().hasHeightForWidth())
        self.label_modelstatus.setSizePolicy(sizePolicy3)
        self.label_modelstatus.setMinimumSize(QSize(250, 0))
        self.label_modelstatus.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_2.addWidget(self.label_modelstatus)

        self.horizontalSpacer = QSpacerItem(36, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.btn_estimate_pose = QPushButton(self.groupBox_2)
        self.btn_estimate_pose.setObjectName(u"btn_estimate_pose")
        sizePolicy1.setHeightForWidth(self.btn_estimate_pose.sizePolicy().hasHeightForWidth())
        self.btn_estimate_pose.setSizePolicy(sizePolicy1)
        icon5 = QIcon()
        icon5.addFile(u"icons/\u7537\u5b692_boy-two.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_estimate_pose.setIcon(icon5)
        self.btn_estimate_pose.setCheckable(True)
        self.btn_estimate_pose.setChecked(False)

        self.horizontalLayout_2.addWidget(self.btn_estimate_pose)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.label_futureframe = QLabel(self.groupBox_2)
        self.label_futureframe.setObjectName(u"label_futureframe")
        sizePolicy2.setHeightForWidth(self.label_futureframe.sizePolicy().hasHeightForWidth())
        self.label_futureframe.setSizePolicy(sizePolicy2)
        self.label_futureframe.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.label_futureframe)

        self.spinBox_futureframe = QSpinBox(self.groupBox_2)
        self.spinBox_futureframe.setObjectName(u"spinBox_futureframe")
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.spinBox_futureframe.sizePolicy().hasHeightForWidth())
        self.spinBox_futureframe.setSizePolicy(sizePolicy4)
        self.spinBox_futureframe.setMinimum(1)
        self.spinBox_futureframe.setMaximum(30)
        self.spinBox_futureframe.setValue(1)

        self.horizontalLayout_2.addWidget(self.spinBox_futureframe)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        sizePolicy2.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy2)

        self.horizontalLayout_2.addWidget(self.label_2)

        self.btn_predict_motion = QPushButton(self.groupBox_2)
        self.btn_predict_motion.setObjectName(u"btn_predict_motion")
        icon6 = QIcon()
        icon6.addFile(u"icons/\u8fd0\u52a8_sport.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_predict_motion.setIcon(icon6)
        self.btn_predict_motion.setCheckable(True)
        self.btn_predict_motion.setChecked(False)

        self.horizontalLayout_2.addWidget(self.btn_predict_motion)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_3)

        self.btn_saveResult = QPushButton(self.groupBox_2)
        self.btn_saveResult.setObjectName(u"btn_saveResult")
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.btn_saveResult.sizePolicy().hasHeightForWidth())
        self.btn_saveResult.setSizePolicy(sizePolicy5)
        icon7 = QIcon()
        icon7.addFile(u"icons/\u4fdd\u5b58_save.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_saveResult.setIcon(icon7)

        self.horizontalLayout_2.addWidget(self.btn_saveResult)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)


        self.gridLayout_5.addWidget(self.groupBox_2, 2, 0, 1, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.plainTextEdit_log = QPlainTextEdit(self.groupBox)
        self.plainTextEdit_log.setObjectName(u"plainTextEdit_log")
        self.plainTextEdit_log.setEnabled(True)
        sizePolicy6 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.plainTextEdit_log.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_log.setSizePolicy(sizePolicy6)

        self.gridLayout.addWidget(self.plainTextEdit_log, 0, 0, 1, 1)


        self.gridLayout_5.addWidget(self.groupBox, 3, 0, 1, 1)

        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(mainWindow)
        self.statusbar.setObjectName(u"statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)

        QMetaObject.connectSlotsByName(mainWindow)
    # setupUi

    def retranslateUi(self, mainWindow):
        mainWindow.setWindowTitle(QCoreApplication.translate("mainWindow", u"\u4eba\u4f53\u8fd0\u52a8\u9884\u6d4b\u53ef\u89c6\u5316\u7cfb\u7edf", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("mainWindow", u"\u8f93\u5165\u89c6\u9891", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("mainWindow", u"\u7ed3\u679c\u89c6\u9891", None))
        self.label.setText(QCoreApplication.translate("mainWindow", u"00:00:00", None))
        self.btn_open.setText(QCoreApplication.translate("mainWindow", u"\u6253\u5f00\u89c6\u9891", None))
        self.btn_play.setText(QCoreApplication.translate("mainWindow", u"\u64ad\u653e", None))
        self.btn_pause.setText(QCoreApplication.translate("mainWindow", u"\u6682\u505c", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("mainWindow", u"\u89c6\u9891\u5904\u7406", None))
        self.label_choosemodel.setText(QCoreApplication.translate("mainWindow", u"\u6a21\u578b\u9009\u62e9\uff1a", None))
        self.comboBox_choosemodel.setItemText(0, QCoreApplication.translate("mainWindow", u"\u8bf7\u9009\u62e9", None))
        self.comboBox_choosemodel.setItemText(1, QCoreApplication.translate("mainWindow", u"AFANAT", None))
        self.comboBox_choosemodel.setItemText(2, QCoreApplication.translate("mainWindow", u"PGBIG", None))
        self.comboBox_choosemodel.setItemText(3, QCoreApplication.translate("mainWindow", u"hmp_ddpm", None))
        self.comboBox_choosemodel.setItemText(4, QCoreApplication.translate("mainWindow", u"ground truth", None))

        self.btn_load_weight.setText(QCoreApplication.translate("mainWindow", u"\u5bfc\u5165\u6743\u91cd", None))
        self.label_modelstatus.setText(QCoreApplication.translate("mainWindow", u"\u672a\u52a0\u8f7d", None))
        self.btn_estimate_pose.setText(QCoreApplication.translate("mainWindow", u"\u4eba\u4f53\u59ff\u6001\u4f30\u8ba1", None))
        self.label_futureframe.setText(QCoreApplication.translate("mainWindow", u"\u672a\u6765\u7b2c", None))
        self.label_2.setText(QCoreApplication.translate("mainWindow", u"\u5e27", None))
        self.btn_predict_motion.setText(QCoreApplication.translate("mainWindow", u"\u4eba\u4f53\u8fd0\u52a8\u9884\u6d4b", None))
        self.btn_saveResult.setText(QCoreApplication.translate("mainWindow", u"\u4fdd\u5b58\u7ed3\u679c\u89c6\u9891", None))
        self.groupBox.setTitle(QCoreApplication.translate("mainWindow", u"\u64cd\u4f5c\u65e5\u5fd7", None))
    # retranslateUi

