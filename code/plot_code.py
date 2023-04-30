import os
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_accln_data(Ax, Ay, Az, imu_ts_unix, title, save=True, save_path=None):
    fig = plt.figure(figsize=(12,10))

    plt.subplot(3, 1, 1)
    plt.plot(imu_ts_unix, Ax)
    plt.grid(linestyle='--')
    plt.title("Ax measurement")

    plt.subplot(3, 1, 2)
    plt.plot(imu_ts_unix, Ay)
    plt.grid(linestyle='--')
    plt.title("Ay measurement")

    plt.subplot(3, 1, 3)
    plt.plot(imu_ts_unix, Az)
    plt.grid(linestyle='--')
    plt.title("Az measurement")
    
    plt.suptitle(title, fontsize=16)
    
    plt.show()
    
    if save == True:
        plt.savefig(save_path)
        
        
        
def plot_gyro_data(Ax, Ay, Az, imu_ts_unix, title, save=True, save_path=None):
    fig = plt.figure(figsize=(12,10))

    plt.subplot(3, 1, 1)
    plt.plot(imu_ts_unix, Ax)
    plt.grid(linestyle='--')
    plt.title("Wx measurement (raw)")

    plt.subplot(3, 1, 2)
    plt.plot(imu_ts_unix, Ay)
    plt.grid(linestyle='--')
    plt.title("Wy measurement (raw)")

    plt.subplot(3, 1, 3)
    plt.plot(imu_ts_unix, Az)
    plt.grid(linestyle='--')
    plt.title("Wz measurement (raw)")
    
    plt.suptitle(title, fontsize=16)
    
    plt.show()
    
    if save == True:
        plt.savefig(save_path)
        
        
        
def plot_est_vs_gt_rpy(vic_rpy_arr, vic_ts_unix, imu_pred_rpy_arr, imu_ts_unix, save=True, save_path=None):
#     vic_ts = vic_ts_unix - vic_ts_unix[0]
#     imu_ts = imu_ts_unix - imu_ts_unix[0]
    vic_ts = vic_ts_unix
    imu_ts = imu_ts_unix
    
    fig = plt.figure(figsize=(12,10))
    
    plt.subplot(3, 1, 1)
    plt.plot(vic_ts, vic_rpy_arr[:, 0], label="True Roll")
    plt.plot(imu_ts, imu_pred_rpy_arr[:, 0], label="Est Roll")
    plt.grid(linestyle='--')
    plt.title("True Roll vs Estimated Roll (in degrees)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(vic_ts, vic_rpy_arr[:, 1], label="True Pitch")
    plt.plot(imu_ts, imu_pred_rpy_arr[:, 1], label="Est Pitch")
    plt.grid(linestyle='--')
    plt.title("True Pitch vs Estimated Pitch (in degrees)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(vic_ts, vic_rpy_arr[:, 2], label="True Yaw")
    plt.plot(imu_ts, imu_pred_rpy_arr[:, 2], label="Est Yaw")
    plt.grid(linestyle='--')
    plt.title("True Yaw vs Estimated Yaw (in degrees)")
    plt.legend()

    plt.show()
    
    if save == True:
        plt.savefig(save_path)
        
        
        
def plot_est_vs_gt_accln(imu_arr_cal, imu_ts_unix, accln_pred_imu_quat, accln_pred_vic_quat, vic_ts_unix, save=True, save_path=None):
#     vic_ts = vic_ts_unix - vic_ts_unix[0]
#     imu_ts = imu_ts_unix - imu_ts_unix[0]
    vic_ts = vic_ts_unix
    imu_ts = imu_ts_unix

    fig = plt.figure(figsize=(12,10))
    
    plt.subplot(3, 1, 1)
    plt.plot(imu_ts, imu_arr_cal[:, 0], label="data")
    plt.plot(imu_ts, accln_pred_imu_quat[:, 0], label="IMU est")
    if accln_pred_vic_quat is not None:
        plt.plot(vic_ts, accln_pred_vic_quat[:, 0], label="VIC est")
    plt.grid(linestyle='--')
    plt.title("Data Ax vs Estimated Ax")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(imu_ts, imu_arr_cal[:, 1], label="data")
    plt.plot(imu_ts, accln_pred_imu_quat[:, 1], label="IMU est")
    if accln_pred_vic_quat is not None:
        plt.plot(vic_ts, accln_pred_vic_quat[:, 1], label="VIC est")
    plt.grid(linestyle='--')
    plt.title("Data Ay vs Estimated Ay")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(imu_ts, imu_arr_cal[:, 2], label="data")
    plt.plot(imu_ts, accln_pred_imu_quat[:, 2], label="IMU est")
    if accln_pred_vic_quat is not None:
        plt.plot(vic_ts, accln_pred_vic_quat[:, 2], label="VIC est")
    plt.grid(linestyle='--')
    plt.title("Data Az vs Estimated Az")
    plt.legend()
    
    plt.show()
    
    if save == True:
        plt.savefig(save_path)
        
        
        
def plot_opt_rpy_traj(opt_rpy_arr, imu_ts_unix, title, save=True, save_path=None):
    fig = plt.figure(figsize=(12,10))

    plt.subplot(3, 1, 1)
    plt.plot(imu_ts_unix, opt_rpy_arr[:,0])
    plt.grid(linestyle='--')
    plt.title("Estimated Optimized Roll Angle (in degrees)")

    plt.subplot(3, 1, 2)
    plt.plot(imu_ts_unix, opt_rpy_arr[:,1])
    plt.grid(linestyle='--')
    plt.title("Estimated Optimized Pitch Angle (in degrees)")

    plt.subplot(3, 1, 3)
    plt.plot(imu_ts_unix, opt_rpy_arr[:,2])
    plt.grid(linestyle='--')
    plt.title("Estimated Optimized Yaw Angle (in degrees)")
    
    plt.suptitle(title, fontsize=16)
    
    plt.show()
    
    if save == True:
        plt.savefig(save_path)