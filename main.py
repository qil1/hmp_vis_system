from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from ui_proj import Ui_mainWindow

from PySide6.QtMultimedia import *
from PySide6.QtMultimediaWidgets import QVideoWidget

from PySide6.QtGui import QColor
from PySide6.QtCore import QTimer, QThread, Signal
from einops import rearrange
import cv2
import os
import pickle
import numpy as np
from utils.draw_pics import draw_pic_single
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import time
import torch
from AFANAT.utils.opt import Options as AFANAT_Options
from AFANAT.models.AFANAT import get_model as get_model_AFANAT
from PGBIG.utils.opt import Options as PGBIG_Options
from PGBIG.models import stage_4
from hmp_ddpm.utils.opt import Options as hmp_ddpm_Options
from hmp_ddpm.models.Predictor import get_model as get_model_hmp_ddpm
from hmp_ddpm.models.Diffusion import DDIMSampler
import shutil
 
# s1. 子线程类：继承自QThread
class WorkThread(QThread):
    # s2. 自定义信号
    signal = Signal(str)

    def __init__(self, filename, buffer_video_id, worker_id, run_model, future_shift=0):
        super().__init__()
        self.filename = filename
        self.worker_id = worker_id
        self.future_shift = future_shift
        self.run_model=run_model
        self.buffer_video_id = buffer_video_id
        self.seq_name = self.filename.split('/')[-1].split('.')[0]
        self.datasetDir = './3dpw_dataset'
        self.file = os.path.join(self.datasetDir,'sequenceFiles',self.seq_name+'.pkl')
        with open(self.file, 'rb') as f:
            self.seq = pickle.load(f, encoding='latin1')
        self.joint_pos = np.array(self.seq['jointPositions'])
        self.tolPerson, self.tolFrame, self.tolDims = self.joint_pos.shape

    # 重写 run() 方法
    def run(self):
        out_fp = './outputs/'+str(self.buffer_video_id)+'.avi'
        if self.worker_id == 0:  # estimate pose
            if self.run_model[0] == 1:  # gt
                plt.ion()
                fig = plt.figure(figsize=(8, 4.5))
                video_out = cv2.VideoWriter(out_fp, cv2.VideoWriter_fourcc(*'XVID'), 30, (800, 450), True)
                for iFrame in range(self.tolFrame):
                    poses = list()
                    for iPerson in range(self.tolPerson):
                        pose = self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[0][iFrame][:3]*1000  # 以第0个人的根结点为坐标原点
                        poses.append(pose)
                    im_arr = cv2.cvtColor(draw_pic_single(fig, poses),cv2.COLOR_RGB2BGR)
                    video_out.write(im_arr)
                    self.signal.emit(str(iFrame))
                plt.close('all')
                video_out.release()

        elif self.worker_id == 1:  # predict motion
            if self.run_model[0] == 1:  # gt
                condition = 10
                shift = self.future_shift
                plt.ion()
                fig = plt.figure(figsize=(8, 4.5))
                video_out = cv2.VideoWriter(out_fp, cv2.VideoWriter_fourcc(*'XVID'), 30, (800, 450), True)
                for iFrame in range(self.tolFrame):
                    poses = list()
                    poses_shift = list()
                    for iPerson in range(self.tolPerson):
                        pose = self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[0][iFrame][:3]*1000  # 以第0个人的根结点为坐标原点
                        poses.append(pose)
                        if iFrame >= condition:
                            if iFrame + shift < self.tolFrame:
                                pose_shift = self.joint_pos[iPerson][iFrame+shift].reshape(-1, 3)*1000 - self.joint_pos[0][iFrame+shift][:3]*1000
                                poses_shift.append(pose_shift)
                            else:
                                poses_shift.append(pose)
                    im_arr = cv2.cvtColor(draw_pic_single(fig, poses, poses_shift),cv2.COLOR_RGB2BGR)
                    video_out.write(im_arr)
                    self.signal.emit(str(iFrame))
                plt.close('all')
                video_out.release()
            elif self.run_model[0] == 2:  # AFANAT
                model = self.run_model[1]
                condition = 10
                t_pred = 30
                joint_num = 23
                shift = self.future_shift
                conditions = [[] for i in range(self.tolPerson)]
                plt.ion()
                fig = plt.figure(figsize=(8, 4.5))
                video_out = cv2.VideoWriter(out_fp, cv2.VideoWriter_fourcc(*'XVID'), 30, (800, 450), True)
                for iFrame in range(self.tolFrame):
                    poses = list()
                    poses_shift = list()
                    for iPerson in range(self.tolPerson):
                        pose = self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[0][iFrame][:3]*1000  # 以第0个人的根结点为坐标原点
                        poses.append(pose)
                        conditions[iPerson].append(self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[iPerson][iFrame][:3]*1000)
                        if len(conditions[iPerson]) > condition:
                            conditions[iPerson] = conditions[iPerson][-condition:]
                        if iFrame >= condition-1:
                            if iFrame + shift < self.tolFrame:
                                inpu = np.array(conditions[iPerson])  # (n_frames, n_joints, n_dims)
                                inpu = torch.tensor(inpu).contiguous()
                                inpu = inpu.unsqueeze(0) / 1000.

                                _, outpu = model(inpu[:, :, 1:, :])
                                outpu = rearrange(outpu, 't b c d -> b t c d')
                                pred = torch.zeros([1, t_pred, joint_num+1, 3])
                                pred[:, :, 1:, :] = outpu
                                pose_shift = pred[:, shift-1, :, :].reshape(-1, 3).detach().numpy()*1000
                                pose_shift[:, :] += self.joint_pos[iPerson][iFrame+shift][:3]*1000 - self.joint_pos[0][iFrame+shift][:3]*1000
                                poses_shift.append(pose_shift)
                            else:
                                poses_shift.append(pose)
                    
                    im_arr = cv2.cvtColor(draw_pic_single(fig, poses, poses_shift),cv2.COLOR_RGB2BGR)
                    video_out.write(im_arr)
                    self.signal.emit(str(iFrame))
                plt.close('all')
                video_out.release()

            elif self.run_model[0] == 3:  # PGBIG
                model = self.run_model[1]
                condition = 10
                t_pred = 30
                joint_num = 23
                shift = self.future_shift
                conditions = [[] for i in range(self.tolPerson)]
                plt.ion()
                fig = plt.figure(figsize=(8, 4.5))
                video_out = cv2.VideoWriter(out_fp, cv2.VideoWriter_fourcc(*'XVID'), 30, (800, 450), True)
                for iFrame in range(self.tolFrame):
                    poses = list()
                    poses_shift = list()
                    for iPerson in range(self.tolPerson):
                        pose = self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[0][iFrame][:3]*1000  # 以第0个人的根结点为坐标原点
                        poses.append(pose)
                        conditions[iPerson].append(self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[iPerson][iFrame][:3]*1000)
                        if len(conditions[iPerson]) > condition:
                            conditions[iPerson] = conditions[iPerson][-condition:]
                        if iFrame >= condition-1:
                            if iFrame + shift < self.tolFrame:
                                inpu = np.array(conditions[iPerson])  # (n_frames, n_joints, n_dims)
                                inpu = torch.tensor(inpu, dtype=torch.float32).contiguous()
                                inpu = inpu.unsqueeze(0)
                                inpu = rearrange(inpu[:, :, 1:, :], 'b t c d -> b t (c d)')

                                p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = model(inpu, input_n=condition, output_n=t_pred, itera=1)
                                p3d_out_all_4 = rearrange(p3d_out_all_4, 'b t (c d) -> b t c d', d=3)
                                pred = torch.zeros([1, t_pred, joint_num+1, 3])
                                pred[:, :, 1:, :] = p3d_out_all_4[:, condition:]
                                pose_shift = pred[:, shift-1, :, :].reshape(-1, 3).detach().numpy()*1000
                                pose_shift[:, :] += self.joint_pos[iPerson][iFrame+shift][:3]*1000 - self.joint_pos[0][iFrame+shift][:3]*1000
                                poses_shift.append(pose_shift)
                            else:
                                poses_shift.append(pose)
                    
                    im_arr = cv2.cvtColor(draw_pic_single(fig, poses, poses_shift),cv2.COLOR_RGB2BGR)
                    video_out.write(im_arr)
                    self.signal.emit(str(iFrame))
                plt.close('all')
                video_out.release()

            elif self.run_model[0] == 4:  # hmp_ddpm
                model = self.run_model[1]
                condition = 10
                t_pred = 30
                joint_num = 23
                shift = self.future_shift
                conditions = [[] for i in range(self.tolPerson)]
                plt.ion()
                fig = plt.figure(figsize=(8, 4.5))
                video_out = cv2.VideoWriter(out_fp, cv2.VideoWriter_fourcc(*'XVID'), 30, (800, 450), True)
                for iFrame in range(self.tolFrame):
                    poses = list()
                    poses_shift = list()
                    for iPerson in range(self.tolPerson):
                        pose = self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[0][iFrame][:3]*1000  # 以第0个人的根结点为坐标原点
                        poses.append(pose)
                        conditions[iPerson].append(self.joint_pos[iPerson][iFrame].reshape(-1, 3)*1000 - self.joint_pos[iPerson][iFrame][:3]*1000)
                        if len(conditions[iPerson]) > condition:
                            conditions[iPerson] = conditions[iPerson][-condition:]
                        if iFrame >= condition-1:
                            if iFrame + shift < self.tolFrame:
                                condition_inp = np.array(conditions[iPerson]) / 1000.
                                condition_inp = torch.tensor(condition_inp[:, 1:, :]).contiguous().unsqueeze(0)  # (n_frames, n_joints, n_dims)
                                noisy_future = torch.randn(size=(1, t_pred, joint_num, 3), device=torch.device('cpu'))
                                sampled_future = model(noisy_future, condition_inp)  # b t c d

                                pred = torch.zeros([1, t_pred, joint_num+1, 3])
                                pred[:, :, 1:, :] = sampled_future
                                
                                pose_shift = pred[:, shift-1, :, :].reshape(-1, 3).detach().numpy()*1000
                                pose_shift[:, :] += self.joint_pos[iPerson][iFrame+shift][:3]*1000 - self.joint_pos[0][iFrame+shift][:3]*1000
                                poses_shift.append(pose_shift)
                            else:
                                poses_shift.append(pose)
                    
                    im_arr = cv2.cvtColor(draw_pic_single(fig, poses, poses_shift),cv2.COLOR_RGB2BGR)
                    video_out.write(im_arr)
                    self.signal.emit(str(iFrame))
                plt.close('all')
                video_out.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_mainWindow()  # UI类的实例化
        self.ui.setupUi(self)
        self.ui.btn_play.setDisabled(True)
        self.ui.btn_pause.setDisabled(True)
        self.ui.btn_estimate_pose.setDisabled(True)
        self.ui.btn_predict_motion.setDisabled(True)
        self.estimate_model_ready = (0, None)
        self.predict_model_ready = (0, None)

        # 播放器
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.ui.widget_video)

        self.player_pred = QMediaPlayer()
        self.player_pred.setVideoOutput(self.ui.widget_video_pred)

        self.timer = QTimer()
        
        # 按钮打开文件
        self.ui.btn_open.clicked.connect(self.openVideoFile)
        # 播放
        self.ui.btn_play.clicked.connect(self.playVideo)  # play
        # 暂停
        self.ui.btn_pause.clicked.connect(self.pauseVideo)  # pause
        self.player.mediaStatusChanged.connect(self.onMediaStatusChanged)
        self.ui.bar_slider.valueChanged.connect(self.onBarSliderValueChanged)

        # 选择模型按钮
        self.ui.comboBox_choosemodel.currentIndexChanged.connect(self.onComboBox_choosemodel_currentIndexChanged)
        self.ui.label_modelstatus.setStyleSheet(f"QLabel {{ color: {QColor('red').name()}; }}")
        self.ui.btn_load_weight.clicked.connect(self.loadModelWeight)
        self.ui.btn_saveResult.clicked.connect(self.onBtnSaveResultClicked)

        self.workThread = None
        self.isworkThreaddeleted = True

        self.ui.btn_estimate_pose.clicked.connect(self.onBtn_estimate_pose_clicked)
        self.ui.btn_predict_motion.clicked.connect(self.onBtn_preidct_motion_clicked)

        self.buffer_video_id = 0
    
    def openVideoFile(self):
        video_name = QFileDialog.getOpenFileName(dir="./3dpw_dataset/Videos")[0]
        if video_name == "":
            return
        self.ui.plainTextEdit_log.appendPlainText("---------------------------------------------------\n打开文件："+video_name)
        self.player.setSource(video_name)
        self.media_duration, self.media_frames = get_duration_from_cv2(video_name)
        self.ui.plainTextEdit_log.appendPlainText("视频时长："+str(self.media_duration)+"秒\n---------------------------------------------------")
        self.ui.btn_play.setDisabled(False)
        self.player.pause()
        self.ui.bar_slider.setMaximum(round(self.media_duration*10))  # 多少个0.1秒
        
        if self.timer:
            self.timer.deleteLater()
        self.timer = QTimer()
        self.time_count = 0
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.onTimerOut)
        
        # 其他清理工作
        self.ui.statusbar.showMessage("")
        self.player_pred.setSource("")

    
    def playVideo(self):
        self.timer.start()
        self.player.play()
        if self.player_pred.isAvailable():
            self.player_pred.play()
        self.ui.btn_play.setDisabled(True)
        self.ui.btn_pause.setDisabled(False)
        self.ui.plainTextEdit_log.appendPlainText("在"+str(self.player.position()/1000.0)+"秒播放")
 
    def pauseVideo(self):
        self.timer.stop()
        self.player.pause()
        if self.player_pred.isAvailable():
            self.player_pred.setPosition(self.player.position())
            self.player_pred.pause()
        self.ui.btn_pause.setDisabled(True)
        self.ui.btn_play.setDisabled(False)
        self.ui.plainTextEdit_log.appendPlainText("在"+str(self.player.position()/1000.0)+"秒暂停")
        
    def onBarSliderValueChanged(self, value):
        if self.player.isPlaying():
            self.timer.stop()
            self.time_count = value
            m, s = divmod(round(self.time_count*0.1), 60)
            h, m = divmod(m, 60)
            self.ui.label.setText("%02d:%02d:%02d" % (h, m, s))
            self.player.setPosition(value*100)
            if self.player_pred.isAvailable():
                self.player_pred.setPosition(value*100)
            self.timer.start()
        else:
            self.time_count = value
            m, s = divmod(round(self.time_count*0.1), 60)
            h, m = divmod(m, 60)
            self.ui.label.setText("%02d:%02d:%02d" % (h, m, s))
            if self.player.isAvailable():
                self.player.setPosition(value*100)
                self.player.pause()
            if self.player_pred.isAvailable():
                self.player_pred.setPosition(value*100)
                self.player_pred.pause()
    
    def onMediaStatusChanged(self):
        self.ui.plainTextEdit_log.appendPlainText("媒体状态："+str(self.player.mediaStatus()))
        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.pause()
            if self.player_pred.isAvailable():
                self.player_pred.setPosition(self.player.position())
                self.player_pred.pause()
            self.ui.btn_pause.setDisabled(True)
            self.ui.btn_play.setDisabled(False)
            self.timer.stop()
            self.time_count = 0

        elif self.player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia:
            if self.estimate_model_ready[0] and self.isworkThreaddeleted:
                self.ui.btn_estimate_pose.setDisabled(False)
            if self.predict_model_ready[0] and self.isworkThreaddeleted:
                self.ui.btn_predict_motion.setDisabled(False)

        else:
            pass

    def onTimerOut(self):
        self.time_count += 1
        self.ui.bar_slider.setValue(round(self.time_count))
        m, s = divmod(round(self.time_count*0.1), 60)
        h, m = divmod(m, 60)
        self.ui.label.setText("%02d:%02d:%02d" % (h, m, s))
        
    def onComboBox_choosemodel_currentIndexChanged(self, curIdx):
        if curIdx != 0:
            self.ui.plainTextEdit_log.appendPlainText(f"选择模型{self.ui.comboBox_choosemodel.itemText(curIdx)}")
        if self.ui.comboBox_choosemodel.itemText(curIdx) == "ground truth":
            self.ui.label_modelstatus.setText("已加载")
            self.ui.label_modelstatus.setStyleSheet(f"QLabel {{ color: {QColor('green').name()}; }}")
            self.estimate_model_ready = (1, None)
            self.predict_model_ready = (1, None)
            if self.player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia and self.isworkThreaddeleted:
                self.ui.btn_estimate_pose.setDisabled(False)
                self.ui.btn_predict_motion.setDisabled(False)
            
        else:
            self.ui.label_modelstatus.setText("未加载")
            self.ui.label_modelstatus.setStyleSheet(f"QLabel {{ color: {QColor('red').name()}; }}")
            self.estimate_model_ready = (0, None)
            self.predict_model_ready = (0, None)
            
            self.ui.btn_estimate_pose.setDisabled(True)
            self.ui.btn_predict_motion.setDisabled(True)

    def loadModelWeight(self):
        curIdx = self.ui.comboBox_choosemodel.currentIndex()
        if self.ui.comboBox_choosemodel.itemText(curIdx) == "AFANAT":
            ckpt_pth = QFileDialog.getOpenFileName(dir="./AFANAT/checkpoints")[0]
            if ckpt_pth == "":
                return
            config = AFANAT_Options().load_config('./AFANAT/configs/3dpw.json')
            device = torch.device('cpu')
            torch.set_default_dtype(torch.float64)
            model = get_model_AFANAT(config, device)
            model.to(device)
            model.eval()
            model_ckpt = pickle.load(open(ckpt_pth, "rb"))
            try:
                model.load_state_dict(model_ckpt['model_dict'])
                self.ui.label_modelstatus.setText(f"{ckpt_pth}")
                self.ui.label_modelstatus.setStyleSheet(f"QLabel {{ color: {QColor('green').name()}; }}")
                self.predict_model_ready = (2, model)
                if self.player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia and self.isworkThreaddeleted:
                    self.ui.btn_predict_motion.setDisabled(False)
                self.ui.plainTextEdit_log.appendPlainText(f"成功导入模型{self.ui.comboBox_choosemodel.itemText(curIdx)}的权重{ckpt_pth}！模型大小：{sum(p.numel() for p in model.parameters()) /1e6}M")
            except:
                self.ui.plainTextEdit_log.appendPlainText(f"导入模型{self.ui.comboBox_choosemodel.itemText(curIdx)}的权重失败！")

        if self.ui.comboBox_choosemodel.itemText(curIdx) == "PGBIG":
            ckpt_pth = QFileDialog.getOpenFileName(dir="./PGBIG/checkpoints")[0]
            if ckpt_pth == "":
                return
            config = PGBIG_Options().load_config('./PGBIG/configs/3dpw.json')
            torch.set_default_dtype(torch.float32)
            net_pred = stage_4.MultiStageModel(opt=config)
            device = torch.device('cpu')
            net_pred.to(device)
            net_pred.eval()
            ckpt = torch.load(ckpt_pth, map_location='cpu')
            try:
                net_pred.load_state_dict(ckpt['state_dict'])
                self.ui.label_modelstatus.setText(f"{ckpt_pth}")
                self.ui.label_modelstatus.setStyleSheet(f"QLabel {{ color: {QColor('green').name()}; }}")
                self.predict_model_ready = (3, net_pred)
                if self.player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia and self.isworkThreaddeleted:
                    self.ui.btn_predict_motion.setDisabled(False)
                self.ui.plainTextEdit_log.appendPlainText(f"成功导入模型{self.ui.comboBox_choosemodel.itemText(curIdx)}的权重{ckpt_pth}！模型大小：{sum(p.numel() for p in net_pred.parameters()) /1e6}M")
            except Exception as e:
                self.ui.plainTextEdit_log.appendPlainText(f"导入模型{self.ui.comboBox_choosemodel.itemText(curIdx)}的权重失败！")
                self.ui.plainTextEdit_log.appendPlainText(f"发生错误：{e}")

        if self.ui.comboBox_choosemodel.itemText(curIdx) == "hmp_ddpm":
            ckpt_pth = QFileDialog.getOpenFileName(dir="./hmp_ddpm/checkpoints")[0]
            if ckpt_pth == "":
                return
            config = hmp_ddpm_Options().load_config('./hmp_ddpm/configs/3dpw.json')
            device = torch.device('cpu')
            torch.set_default_dtype(torch.float64)
            
            model = get_model_hmp_ddpm(config, device)
            model_cp = pickle.load(open(ckpt_pth, "rb"))
    
            try:
                model.load_state_dict(model_cp['model_dict'])
                model.eval()
                sampler = DDIMSampler(
                    model, config.beta_1, config.beta_T, config.T, w=config.w, device=device).to(device)
                sampler.eval()
                self.ui.label_modelstatus.setText(f"{ckpt_pth}")
                self.ui.label_modelstatus.setStyleSheet(f"QLabel {{ color: {QColor('green').name()}; }}")
                self.predict_model_ready = (4, sampler)
                if self.player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia and self.isworkThreaddeleted:
                    self.ui.btn_predict_motion.setDisabled(False)
                self.ui.plainTextEdit_log.appendPlainText(f"成功导入模型{self.ui.comboBox_choosemodel.itemText(curIdx)}的权重{ckpt_pth}！模型大小：{sum(p.numel() for p in sampler.parameters()) /1e6}M")
            except Exception as e:
                self.ui.plainTextEdit_log.appendPlainText(f"导入模型{self.ui.comboBox_choosemodel.itemText(curIdx)}的权重失败！")
                self.ui.plainTextEdit_log.appendPlainText(f"发生错误：{e}")


    def onBtn_estimate_pose_clicked(self):
        # TODO: button disabled
        self.ui.btn_open.setDisabled(True)
        self.ui.btn_estimate_pose.setDisabled(True)
        self.ui.btn_predict_motion.setDisabled(True)

        # s4. 实例化子线程
        self.buffer_video_id = (self.buffer_video_id + 1) % 2
        self.workThread = WorkThread(self.player.source().toString(), self.buffer_video_id, worker_id=0, run_model=self.estimate_model_ready)  # 用self实例化，防止子线程被回收
        # s5. 对子线程的信号进行绑定
        self.workThread.signal.connect(lambda x: self.ui.statusbar.showMessage(f"正在估计视频人体姿态...当前进度为：{100.0*(int(x)+1)/self.media_frames:.2f}% ({int(x)+1}/{int(self.media_frames)})"))  # 将信号连接到标签
        self.workThread.started.connect(lambda: self.ui.statusbar.showMessage("子线程开启"))         # 子线程开启时激活
        self.workThread.finished.connect(self.onEstimateWorkThreadFinished)        # 子线程结束时激活
        self.workThread.finished.connect(lambda: self.workThread.deleteLater())         # 释放子线程
        # s6. 开启子线程
        self.workThread_st = time.time()
        self.ui.plainTextEdit_log.appendPlainText(f"使用模型{self.ui.comboBox_choosemodel.itemText(self.ui.comboBox_choosemodel.currentIndex())}估计人体姿态")
        self.workThread.start()
        self.isworkThreaddeleted = False

    def onBtn_preidct_motion_clicked(self):
        self.ui.btn_open.setDisabled(True)
        self.ui.btn_estimate_pose.setDisabled(True)
        self.ui.btn_predict_motion.setDisabled(True)

        future_shift = self.ui.spinBox_futureframe.value()
        self.buffer_video_id = (self.buffer_video_id + 1) % 2
        self.workThread = WorkThread(self.player.source().toString(), self.buffer_video_id, worker_id=1, run_model=self.predict_model_ready, future_shift=future_shift)
        self.workThread.signal.connect(lambda x: self.ui.statusbar.showMessage(f"正在预测视频未来{future_shift}帧人体运动...当前进度为：{100.0*(int(x)+1)/self.media_frames:.2f}% ({int(x)+1}/{int(self.media_frames)})"))  # 将信号连接到标签
        self.workThread.started.connect(lambda: self.ui.statusbar.showMessage("子线程开启"))         # 子线程开启时激活
        self.workThread.finished.connect(self.onPredictWorkThreadFinished)        # 子线程结束时激活
        self.workThread.finished.connect(lambda: self.workThread.deleteLater())         # 释放子线程
        # s6. 开启子线程
        self.workThread_st = time.time()
        self.ui.plainTextEdit_log.appendPlainText(f"使用模型{self.ui.comboBox_choosemodel.itemText(self.ui.comboBox_choosemodel.currentIndex())}预测未来{future_shift}帧运动")
        self.workThread.start()
        self.isworkThreaddeleted = False

    def onEstimateWorkThreadFinished(self):
        self.workThread_ed = time.time()
        # TODO: button recover
        self.ui.btn_open.setDisabled(False)
        self.ui.btn_estimate_pose.setDisabled(False)
        if self.predict_model_ready[0]:
            self.ui.btn_predict_motion.setDisabled(False)

        self.ui.plainTextEdit_log.appendPlainText(f"人体姿态估计完成，共耗时{self.workThread_ed-self.workThread_st:.2f}s")
        self.ui.statusbar.showMessage("人体姿态估计完成！子线程关闭")

        out_video_fp = './outputs/' + str(self.buffer_video_id) + '.avi'
        self.player_pred.setSource(out_video_fp)
        self.player_pred.setPosition(self.player.position())
        self.player_pred.pause()

        self.isworkThreaddeleted = True

    def onPredictWorkThreadFinished(self):
        self.workThread_ed = time.time()
        # TODO: button recover
        self.ui.btn_open.setDisabled(False)
        self.ui.btn_predict_motion.setDisabled(False)
        if self.estimate_model_ready[0]:
            self.ui.btn_estimate_pose.setDisabled(False)

        self.ui.plainTextEdit_log.appendPlainText(f"人体运动预测完成，共耗时{self.workThread_ed-self.workThread_st:.2f}s")
        self.ui.statusbar.showMessage("人体运动预测完成！子线程关闭")

        out_video_fp = './outputs/' + str(self.buffer_video_id) + '.avi'
        self.player_pred.setSource(out_video_fp)
        self.player_pred.setPosition(self.player.position())
        self.player_pred.pause()
        
        self.isworkThreaddeleted = True
        
    def onBtnSaveResultClicked(self):
        if self.player_pred.mediaStatus() != QMediaPlayer.MediaStatus.BufferedMedia:
            return
        vfp = self.player_pred.source().toString()
        if not vfp:
            return
        
        tfp, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Video Files (*.avi);;ALL Files (*)")
        if tfp:
            try:
                shutil.copy(vfp, tfp)
                QMessageBox.information(self, "保存成功！", "Save file {} successfully!".format(tfp))
            except IOError:
                QMessageBox.warning(self, "保存失败", "Cannot write file {}".format(tfp))


def get_duration_from_cv2(filename):
        cap = cv2.VideoCapture(filename)
        if cap.isOpened():
            rate = cap.get(5)
            frame_num =cap.get(7)
            duration = frame_num/rate
            return duration, frame_num
        return -1


if __name__ == '__main__':
    app = QApplication([])  # 启动一个应用
    window = MainWindow()  # 实例化主窗口
    window.show()  # 展示主窗口
    app.exec()  # 避免程序执行到这一行后直接退出