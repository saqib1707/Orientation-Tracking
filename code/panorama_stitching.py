import os
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import transforms3d as t3d

# import numpy as np
import autograd
import autograd.numpy as np

import load_data
from rotplot import rotplot
from orientation_tracking import load_dataset


def conv_sph2cart(theta_long, theta_lat, radius=1.0):
    assert(theta_long.shape == theta_lat.shape)
    height, width = theta_lat.shape

    cart_coords = np.zeros((height, width, 3))
    cart_coords[:,:,0] = radius * np.cos(theta_lat) * np.cos(theta_long)
    cart_coords[:,:,1] = radius * np.cos(theta_lat) * np.sin(theta_long)
    cart_coords[:,:,2] = -1 * radius * np.sin(theta_lat)

    return cart_coords


def conv_cart2sph(cart_coords, radius=1):
    X, Y, Z = cart_coords[:,:,0], cart_coords[:,:,1], cart_coords[:,:,2]
    
    theta_lat = np.arcsin(-Z / radius)
    theta_long = np.arctan2(Y, X)
    
    return theta_long, theta_lat


def find_camvic_closest_ts(vic_ts_unix, ts):
    tmp = ts - vic_ts_unix
    idx = np.where(tmp > 0)[0][-1]
    return idx


def estimate_stitched_img(cam_arr, cam_ts_unix, vic_arr, vic_ts_unix):
    num_cam_ts = cam_ts_unix.shape[0]
    num_imgs, img_height, img_width, _ = cam_arr.shape
    assert(num_imgs == num_cam_ts)
    center_u, center_v = img_height//2, img_width//2
    print("Number of Images: {}, Image (H,W): ({},{})".format(num_imgs, img_height, img_width))

    # specify the horizontal and vertical FOV in degrees
    hor_fov = 60
    ver_fov = 45

    # specify the panorama output image size
    pan_img_height, pan_img_width = 720, 1080
    pan_img = np.zeros((pan_img_height+1, pan_img_width+1, 3), dtype=np.uint8)

    V, U = np.meshgrid(np.arange(0,img_width), np.arange(0,img_height))

    # compute spherical coordinates # (in degrees) assuming linear scale
    theta_long_deg = -(V - center_v) * hor_fov / img_width
    theta_lat_deg = (U - center_u) * ver_fov / img_height

    # convert degrees to radians
    theta_long = theta_long_deg * np.pi / 180.0
    theta_lat = theta_lat_deg * np.pi / 180.0

    # convert spherical to cartesian coordinates assuming radius/depth=1
    cart_coords = conv_sph2cart(theta_long, theta_lat, radius=1)    # (240, 320, 3)

    for t in range(num_cam_ts):
        # find the closest-in-the-past timestamp of orientation to each camera image
        vic_closest_idx = find_camvic_closest_ts(vic_ts_unix, cam_ts_unix[t])
        rot_mat_gt = vic_arr[vic_closest_idx]

        # rotate cartesian coordinates to world frame using VICON ground-truth R
        cart_coords_rot = np.matmul(cart_coords, rot_mat_gt.T)    # (240, 320, 3)

        # convert cartesian back to spherical coordinates
        theta_long_rot, theta_lat_rot = conv_cart2sph(cart_coords_rot)  # (240, 320), (240, 320)

        pan_pix_coords_u = ((theta_lat_rot + np.pi/2) / np.pi) * pan_img_height   # (240, 320)
        pan_pix_coords_v = ((np.pi - theta_long_rot) / (2*np.pi)) * pan_img_width   # (240, 320)

        pan_pix_coords_u = np.round(pan_pix_coords_u).astype(np.int32)
        pan_pix_coords_v = np.round(pan_pix_coords_v).astype(np.int32)
        
        if np.max(pan_pix_coords_u) > pan_img_height or np.max(pan_pix_coords_u) < 0 or np.min(pan_pix_coords_u) < 0 or np.min(pan_pix_coords_u) > pan_img_height:
            continue
        if np.max(pan_pix_coords_v) > pan_img_width or np.max(pan_pix_coords_v) < 0 or np.min(pan_pix_coords_v) < 0 or np.min(pan_pix_coords_v) > pan_img_width:
            continue

        pan_img[pan_pix_coords_u, pan_pix_coords_v] = np.copy(cam_arr[t])
    
    return pan_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify options")
    parser.add_argument('--dataset_idx', type=int, default=1, metavar='N', help='Dataset Index')
    args = parser.parse_args()

    cam_arr, cam_ts_unix, imu_arr, imu_ts_unix, vic_arr, vic_ts_unix = load_dataset(args.dataset_idx)
    
    if cam_arr is None:
        print("No camera images available --> Stitching not possible !!!")

    # load the optimized quaternion trajectory array estimated using orientation tracking code
    opt_quat_arr = np.load("../data/ckpt_weights/dataset_"+str(args.dataset_idx)+"/opt_quat_arr.npy")
    print("Orientation trajectory loaded !!!")
    
    # convert optimized quaternions to rotation matrices
    assert(opt_quat_arr.shape[0] == imu_arr.shape[0])
    opt_rot_mat = np.zeros((opt_quat_arr.shape[0], 3, 3), dtype=np.float32)
    for t in range(opt_quat_arr.shape[0]):
        opt_rot_mat[t] = np.array(t3d.quaternions.quat2mat(opt_quat_arr[t]), dtype=np.float32)
    
    use_vic_rot = True
    if args.dataset_idx == 10 or args.dataset_idx == 11:
        # use optimized trajectory for estimating panorama image
        pan_img = estimate_stitched_img(cam_arr, cam_ts_unix, opt_rot_mat, imu_ts_unix)
    else:
        if use_vic_rot == True:
            pan_img = estimate_stitched_img(cam_arr, cam_ts_unix, vic_arr, vic_ts_unix)
        else:
            pan_img = estimate_stitched_img(cam_arr, cam_ts_unix, opt_rot_mat, imu_ts_unix)

    if use_vic_rot == True:
        fig = plt.figure(figsize=(15,10))
        plt.imshow(pan_img)
        plt.savefig("../plots/panorama_stitching/pan_img_vic_ds_"+str(args.dataset_idx)+".png")
    else:
        fig = plt.figure(figsize=(15,10))
        plt.imshow(pan_img)
        plt.savefig("../plots/panorama_stitching/pan_img_imu_ds_"+str(args.dataset_idx)+".png")