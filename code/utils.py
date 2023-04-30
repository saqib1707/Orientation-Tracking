# import numpy as np
import math
import autograd.numpy as np


def compute_exp_quat_mat(q):
#     assert(q.shape[1] == 4)
    
    qv_mag = np.linalg.norm(q[:, 1:], axis=1)
    idx = np.where(qv_mag > 0)[0]
    
    exp_q = np.zeros_like(q)
    exp_q[:, 0] = np.exp(q[:, 0]) * np.cos(qv_mag)
    exp_q[idx, 1:] = np.expand_dims(np.exp(q[idx, 0]) * np.sin(qv_mag[idx]), axis=1) * (q[idx, 1:] / np.expand_dims(qv_mag[idx], axis=1))
    
    return exp_q


def compute_quat_inv_mat(q):
#     assert(q.shape[1] == 4)

    q_inv = np.array([q[:,0], -q[:,1], -q[:,2], -q[:,3]]).T
    q_inv = q_inv / np.expand_dims(np.square(np.linalg.norm(q, axis=1)), axis=1)
    
    return q_inv


def compute_quat_prod_mat(q, p):
#     assert(q.shape[1] == p.shape[1] == 4)
    
    first_col = q[:,0] * p[:,0] - np.sum(q[:,1:] * p[:,1:], axis=1)
    sec_to_four_col = np.expand_dims(q[:,0], axis=1) * p[:,1:] + np.expand_dims(p[:,0], axis=1) * q[:,1:] + np.cross(q[:,1:], p[:,1:])
    
    quat_prod = np.array([first_col, sec_to_four_col[:,0], sec_to_four_col[:,1], sec_to_four_col[:,2]]).T
    return quat_prod


def compute_log_quat_mat(q):
#     assert(q.shape[1] == 4)
    
    eps = 1e-10
    noise = np.random.rand(q.shape[0]) * eps
    
    qv_mag = np.linalg.norm(q[:, 1:], axis=1) + noise
    q_mag = np.linalg.norm(q, axis=1) + noise
#     idx = np.where(qv_mag == 0)[0]

    first_col = np.log(q_mag)
    sec_to_four_col = np.expand_dims(np.arccos(q[:, 0] / q_mag), axis=1) * q[:, 1:] / np.expand_dims(qv_mag, axis=1)
    
    log_q = np.array([first_col, sec_to_four_col[:,0], sec_to_four_col[:,1], sec_to_four_col[:,2]])
    return log_q


def compute_f_mat(q, tau, omega):
#     assert(q.shape[0] == omega.shape[0] == tau.shape[0])
    
    quat_omega = np.zeros_like(q)
    quat_omega[:, 1:] = tau * omega / 2
    exp_quat_omega = compute_exp_quat_mat(quat_omega)
    f_out = compute_quat_prod_mat(q, exp_quat_omega)
    
    return f_out


def compute_h_mat(q, GRAVITY):
    tmp = np.zeros_like(q)
    tmp[:, 3] = -GRAVITY

    return compute_quat_prod_mat(compute_quat_prod_mat(compute_quat_inv_mat(q), tmp), q)


def compute_exp_quat(q):
    qv_mag = np.linalg.norm(q[1:])
    
    if qv_mag == 0:
        exp_q = np.array([np.exp(q[0]), 0, 0, 0], dtype=np.float32)
    else:
        tmp = (q[1:] / qv_mag) * np.sin(qv_mag)
        exp_q = np.exp(q[0]) * np.array([np.cos(qv_mag), tmp[0], tmp[1], tmp[2]])
    
    return exp_q


def compute_log_quat(q):
    qv_mag = np.linalg.norm(q[1:])
    
    if qv_mag == 0:
        log_q = np.array([np.log(np.abs(q[0])), 0, 0, 0], dtype=np.float32)
    else:
        q_mag = np.linalg.norm(q)
        tmp = (q[1:] / qv_mag) * np.arccos(q[0] / q_mag)
        log_q = np.array([np.log(q_mag), tmp[0], tmp[1], tmp[2]])
    
    return log_q


def compute_quat_inv(q):
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    q_inv = q_inv / np.linalg.norm(q)**2
    
    return q_inv


def compute_quat_prod(q, p):
#     assert(q.shape[0] == p.shape[0] == 4)
    tmp = np.array([q[0] * p[1:] + p[0] * q[1:] + np.cross(q[1:], p[1:])])
    quat_prod = np.array([q[0] * p[0] - np.dot(q[1:], p[1:]), tmp[0,0], tmp[0,1], tmp[0,2]])
    
    return quat_prod


def compute_f(q, tau, omega):
    quat_omega = np.insert(tau * omega / 2, 0, 0)
    exp_quat_omega = compute_exp_quat(quat_omega)
    f_out = compute_quat_prod(q, exp_quat_omega)
    
    return f_out


def compute_h(q, GRAVITY):
    tmp = np.array([0, 0, 0, -GRAVITY], dtype=np.float32)
    return compute_quat_prod(compute_quat_prod(compute_quat_inv(q), tmp), q)