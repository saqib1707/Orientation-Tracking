import os
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import transforms3d as t3d

# import jax.numpy as jnp
# from jax import grad, jacrev
# from jax import random

import autograd
import autograd.numpy as np
# import numpy as np

import load_data
from rotplot import rotplot
from utils import *
from plot_code import *


# Define Constants
GRAVITY = -1.0


def load_dataset(dataset_idx):
    start_time = time.time()
    
    if dataset_idx >= 1 and dataset_idx <= 9:
        setname = "trainset"
    elif dataset_idx >= 10 and dataset_idx <= 11:
        setname = "testset"
    else:
        print("Invalid dataset index !!!")
        return

    if dataset_idx == 1 or dataset_idx == 2 or (dataset_idx >= 8 and dataset_idx <=11):
        cfile = "../data/"+setname+"/cam/cam" + str(dataset_idx) + ".p"
        camd = load_data.read_data(cfile)
    else:
        camd = None

    ifile = "../data/"+setname+"/imu/imuRaw" + str(dataset_idx) + ".p"
    imud = load_data.read_data(ifile)

    if setname == "trainset":
        vfile = "../data/"+setname+"/vicon/viconRot" + str(dataset_idx) + ".p"
        vicd = load_data.read_data(vfile)
    else:
        vicd = None
    
    print("Raw Data:")
    if camd is not None:
        cam_arr, cam_ts_unix = np.transpose(camd['cam'], (3, 0, 1, 2)), camd['ts'].T
        print("Camera data - Image shape: {}, Timestamp shape: {}".format(cam_arr.shape, cam_ts_unix.shape))
    else:
        cam_arr, cam_ts_unix = None, None

    imu_arr, imu_ts_unix = np.array(imud['vals'].T, dtype=np.float32), imud['ts'].T
    imu_arr[:, [5, 3, 4]] = imu_arr[:, [3, 4, 5]]         # (Ax, Ay, Az, Wx, Wy, Wz)
    print("IMU data - IMU shape: {}, Timestamp shape: {}".format(imu_arr.shape, imu_ts_unix.shape))
    
    if vicd is not None:
        vic_arr, vic_ts_unix = np.transpose(vicd['rots'], (2, 0, 1)), vicd['ts'].T
        print("VICON data - IMU shape: {}, Timestamp shape: {}".format(vic_arr.shape, vic_ts_unix.shape))
    else:
        vic_arr, vic_ts_unix = None, None
    
    print('%s took: %s sec.\n' % ("Data import", round(time.time() - start_time, 4)))
    
    return cam_arr, cam_ts_unix, imu_arr, imu_ts_unix, vic_arr, vic_ts_unix


def calibrate_IMU_data(imu_arr, num_init_samp_bias=500):
    Vref = 3300.0
    
    # compute sensitivities of accelerometer and gyroscope
    sens_gyro = 3.33 / (np.pi/180)
    sens_acclm = 300 / np.abs(GRAVITY)
    print("Sensitivity: Acclm =", sens_acclm, "mV/m/sec2,  Gyro =", sens_gyro, "mV/rad/sec")

    # compute scale factor of accelerometer and gyroscope
    scale_gyro = Vref / (1023 * sens_gyro)
    scale_acclm = Vref / (1023 * sens_acclm)
    print("Scale: Acclm = ", scale_acclm, "m/sec2,  Gyro =", scale_gyro, "rad-sec\n")

    # estimate accelerometer bias
    imu_acclm_bias = np.mean(imu_arr[0:num_init_samp_bias, 0:3], axis=0)
    imu_acclm_bias_gravity = np.abs(GRAVITY) / scale_acclm
    imu_acclm_bias[2] = imu_acclm_bias[2] - imu_acclm_bias_gravity
    print("Accelerometer bias:", imu_acclm_bias)

    # estimate gyrometer bias
    imu_gyro_bias = np.mean(imu_arr[0:num_init_samp_bias, 3:6], axis=0)
    print("Gyroscope bias:", imu_gyro_bias, "\n")
    
    imu_arr_cal = np.zeros_like(imu_arr)
    imu_arr_cal[:, 0:3] = (imu_arr[:, 0:3] - imu_acclm_bias) * scale_acclm
    imu_arr_cal[:, 0:2] = imu_arr_cal[:, 0:2] * -1.0
    imu_arr_cal[:, 3:6] = (imu_arr[:, 3:6] - imu_gyro_bias) * scale_gyro

    return imu_arr_cal


def est_accln_rot_vic_data(imu_arr_cal, imu_ts_unix, vic_arr):
    # estimate true euler rotation angles from VICON data
    num_vic_ts = vic_arr.shape[0]
    vic_rpy_arr = np.zeros((num_vic_ts, 3), dtype=np.float32)
    vic_quat_arr = np.zeros((num_vic_ts, 4), dtype=np.float32)

    for t in range(num_vic_ts):
        rot_mat = vic_arr[t]
        vic_rpy_arr[t, :] = np.array(t3d.euler.mat2euler(rot_mat)) * 180 / np.pi
#         vic_quat_arr[t, :] = np.array(t3d.quaternions.mat2quat(rot_mat))

    # compute acceleration using the VICON quaternion data and observation model
    accln_pred_vic_quat = None
#     accln_pred_vic_quat = compute_h_mat(vic_quat_arr, GRAVITY)[:, 1:]

    # estimate predicted euler angles from IMU data using quaternion kinematics motion model
    num_imu_ts = imu_arr_cal.shape[0]
    quat_pred = np.zeros((num_imu_ts, 4), dtype=np.float32)
    quat_pred[0] = np.array([1, 0, 0, 0], dtype=np.float32)   # define q0

    imu_pred_rpy_arr = np.zeros((num_imu_ts, 3), dtype=np.float32)
    imu_pred_rpy_arr[0] = np.array(t3d.euler.quat2euler(quat_pred[0], axes='sxyz')) * 180 / np.pi

    for t in range(num_imu_ts - 1):
        tau_t = imu_ts_unix[t+1] - imu_ts_unix[t]
        omega_t = imu_arr_cal[t, 3:6]
        quat_pred[t+1] = compute_f(quat_pred[t], tau_t, omega_t)

        imu_pred_rpy_arr[t+1, :] = np.array(t3d.euler.quat2euler(quat_pred[t+1], axes='sxyz')) * 180 / np.pi

    # compute acceleration using the IMU quaternion data and observation model
    accln_pred_imu_quat = compute_h_mat(quat_pred, GRAVITY)[:, 1:]
    
    return vic_rpy_arr, imu_pred_rpy_arr, accln_pred_vic_quat, accln_pred_imu_quat

    
def compute_cost_func_mat(quat_opt_var):
    eps = 1e-10
    noise = np.random.rand(quat_opt_var.shape[0], quat_opt_var.shape[1]) * eps

    q0 = np.array([1, 0, 0, 0], dtype=np.float32)
    cost = 0
    T = quat_opt_var.shape[0]
    
    q_inv_mat = compute_quat_inv_mat(quat_opt_var)
    h_mat = compute_h_mat(quat_opt_var, GRAVITY)[:, 1:]

    tmp = np.vstack([q0, quat_opt_var[0:-1]])
    
    tau_mat = imu_ts_unix[1:] - imu_ts_unix[0:-1]
    omega_mat = imu_arr_cal[:-1, 3:6]

    f_out = compute_f_mat(tmp, tau_mat, omega_mat)
    
    tmp1 = compute_quat_prod_mat(q_inv_mat, f_out) + noise
    tmp2 = 2 * compute_log_quat_mat(tmp1)
    
    tmp3 = np.sum(np.square(np.linalg.norm(tmp2, axis=1)))
    tmp4 = np.sum(np.square(np.linalg.norm(imu_arr_cal[1:, 0:3] - h_mat, axis=1)))
    
    cost = 0.5 * (tmp3 + tmp4)

    return cost


def est_quat_traj_using_PGD(imu_arr_cal, imu_ts_unix, num_itr=50, lr=1e-2, save=True, save_path=None):
    q0 = np.array([1, 0, 0, 0], dtype=np.float32)

    # estimate predicted euler angles from IMU data using quaternion kinematics motion model
    num_imu_ts = imu_arr_cal.shape[0]    # 5645
    T = num_imu_ts - 1                   # 5644

    # initialize quaternion optimization variables q1:T
    quat_opt_var = np.zeros((T, 4), dtype=np.float32)
    
    for t in range(0, T):
        tau_t = imu_ts_unix[t+1] - imu_ts_unix[t]
        omega_t = imu_arr_cal[t, 3:6]
        quat_opt_var[t] = compute_f(q0, tau_t, omega_t) if t == 0 else compute_f(quat_opt_var[t-1], tau_t, omega_t)

    cost_lst = [compute_cost_func_mat(quat_opt_var)]

    grad_cost_fn = autograd.grad(compute_cost_func_mat)
#     grad_cost_fn = autograd.jacobian(compute_cost_func_mat)

    print("Projected Gradient Descent Start")
    start_time = time.time()
    for itr in range(num_itr):
        print("Iteration:", itr)
        if itr == num_itr//2:
            lr = lr/5
        quat_opt_var = quat_opt_var - lr * grad_cost_fn(quat_opt_var)
        quat_opt_var = quat_opt_var / np.reshape(np.linalg.norm(quat_opt_var, axis=1), (T, 1))

        cost_lst.append(compute_cost_func_mat(quat_opt_var))

    print("Time taken: {} sec".format(round(time.time() - start_time, 5)))
    print("Projected Gradient Descent End")

    fig = plt.figure(figsize=(10, 7))
    plt.plot(cost_lst)
    plt.grid(linestyle='--')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost c(q1:T)", fontsize=14)
    plt.title("Cost Function", fontsize=16)
    plt.show()
    
    if save == True:
        plt.savefig(save_path)
        
    return quat_opt_var
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify options")
    parser.add_argument('--dataset_idx', type=int, default=1, metavar='N', help='Dataset Index')
    parser.add_argument('--num_pgd_itr', type=int, default=500, metavar='N', help='Number of PGD iterations')
    args = parser.parse_args()
    
    cam_arr, cam_ts_unix, imu_arr, imu_ts_unix, vic_arr, vic_ts_unix = load_dataset(args.dataset_idx)
    os.makedirs("../plots/orientation_tracking/dataset_"+str(args.dataset_idx), exist_ok=True)
    
    num_imu_ts = imu_ts_unix.shape[0]

    # plot raw acceleration data
    print("Plotting and saving raw accelerometer data")
    num_samp_plot = imu_arr.shape[0]
    plt_title = "Raw Accelerometer Measurements"
    save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/raw_accln_data.png"
    plot_accln_data(imu_arr[0:num_samp_plot, 0], imu_arr[0:num_samp_plot, 1], imu_arr[0:num_samp_plot, 2], imu_ts_unix, plt_title, True, save_path)
    
    # plot raw gyroscope data
    print("Plotting and saving raw gyroscope data\n")
    num_samp_plot = imu_arr.shape[0]
    plt_title = "Raw Gyroscope Measurements"
    save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/raw_gyro_data.png"
    plot_gyro_data(imu_arr[0:num_samp_plot, 3], imu_arr[0:num_samp_plot, 4], imu_arr[0:num_samp_plot, 5], imu_ts_unix, plt_title, True, save_path)
    
    # calibrate IMU data
    if args.dataset_idx == 6 or args.dataset_idx == 7 or args.dataset_idx == 10:
        num_init_samp_bias = 400
    elif args.dataset_idx == 5 or args.dataset_idx == 11:
        num_init_samp_bias = 350
    else:
        num_init_samp_bias = 500

    imu_arr_cal = calibrate_IMU_data(imu_arr, num_init_samp_bias)
    
    # plot calibrated acceleration data
    print("Plotting and saving calibrated accelerometer data")
    num_samp_plot = imu_arr_cal.shape[0]
    plt_title = "Calibrated Accelerometer Measurements"
    save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/cal_accln_data.png"
    plot_accln_data(imu_arr_cal[0:num_samp_plot, 0], imu_arr_cal[0:num_samp_plot, 1], imu_arr_cal[0:num_samp_plot, 2], imu_ts_unix, plt_title, True, save_path)
    
    # plot calibrated gyroscope data
    print("Plotting and saving calibrated gyroscope data\n")
    num_samp_plot = imu_arr_cal.shape[0]
    plt_title = "Calibrated Gyroscope Measurements"
    save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/cal_gyro_data.png"
    plot_gyro_data(imu_arr_cal[0:num_samp_plot, 3], imu_arr_cal[0:num_samp_plot, 4], imu_arr_cal[0:num_samp_plot, 5], imu_ts_unix, plt_title, True, save_path)
    
    
    # verify the estimated orientation and acceleration data using motion and observation model
    if args.dataset_idx >= 1 and args.dataset_idx <= 9:
        vic_rpy_arr, imu_pred_rpy_arr, accln_pred_vic_quat, accln_pred_imu_quat = est_accln_rot_vic_data(imu_arr_cal, imu_ts_unix, vic_arr)
        
        # plot estimated RPY angles to the ground-truth RPY angles
        save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/est_vs_gt_rpy.png"
        plot_est_vs_gt_rpy(vic_rpy_arr, vic_ts_unix, imu_pred_rpy_arr, imu_ts_unix, save=True, save_path=save_path)

        save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/est_vs_gt_accln.png"
        plot_est_vs_gt_accln(imu_arr_cal, imu_ts_unix, accln_pred_imu_quat, accln_pred_vic_quat, vic_ts_unix, save=True, save_path=save_path)


    # Projected Gradient Descent Training
    save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/PGD_cost_vs_itr.png"
    quat_opt_var = est_quat_traj_using_PGD(imu_arr_cal, imu_ts_unix, num_itr=args.num_pgd_itr, lr=5e-3, save=True, save_path=save_path)
    
    # estimate the RPY angles using the estimated optimized trajectory
    q0 = np.array([1,0,0,0], dtype=np.float32)
    quat_opt_var = np.vstack([q0, quat_opt_var])
    
    # save the optimized orientation trajectory in a numpy file
    os.makedirs("../data/ckpt_weights/", exist_ok=True)
    os.makedirs("../data/ckpt_weights/dataset_"+str(args.dataset_idx), exist_ok=True)
    np.save("../data/ckpt_weights/dataset_"+str(args.dataset_idx)+"/opt_quat_arr.npy", quat_opt_var)
    
    opt_rpy_arr = np.zeros((num_imu_ts, 3), dtype=np.float32)
    for t in range(num_imu_ts):
        opt_rpy_arr[t] = np.array(t3d.euler.quat2euler(quat_opt_var[t], axes='sxyz')) * 180 / np.pi
    
    if args.dataset_idx >= 1 and args.dataset_idx <= 9:
        save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/opt_vs_gt_rpy.png"
        plot_est_vs_gt_rpy(vic_rpy_arr, vic_ts_unix, opt_rpy_arr, imu_ts_unix, save=True, save_path=save_path)
    elif args.dataset_idx >= 10 and args.dataset_idx <= 11:
        plt_title = "Optimized Orientation Trajectory on test data"
        save_path = "../plots/orientation_tracking/dataset_"+str(args.dataset_idx)+"/opt_rpy_test.png"
        plot_opt_rpy_traj(opt_rpy_arr, imu_ts_unix, plt_title, True, save_path)