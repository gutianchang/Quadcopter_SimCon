import numpy as np
import matplotlib.pyplot as plt

def ekf(dt, z, q, r, x_aposteriori_k, P_aposteriori_k):
    w_m = z[0:3]
    a_m = z[3:6]
    g = np.sum(np.array(a_m)**2)

    w_x_ = [[0,                              -(w_m[2] - x_aposteriori_k[2]), w_m[1] - x_aposteriori_k[1]   ],
            [w_m[2] - x_aposteriori_k[2],    0,                              -(w_m[0] - x_aposteriori_k[0])],
            [-(w_m[1] - x_aposteriori_k[1]), w_m[0] - x_aposteriori_k[0],    0                             ]]

    w_x_ = np.mat(w_x_)

    bCn = np.eye(3) - w_x_ * dt
    x_apriori = np.zeros([6, 1])
    x_apriori[0:3, 0]  = x_aposteriori_k[0:3]   
    x_apriori[3:6] = bCn * np.mat(x_aposteriori_k[3:6]).T

    x_aposteriori_k_x = [[0,                    -x_aposteriori_k[5], x_aposteriori_k[4] ],
                         [x_aposteriori_k[5],   0,                   -x_aposteriori_k[3]],
                         [-x_aposteriori_k[4],  x_aposteriori_k[3],  0                  ]]

    x_aposteriori_k_x = np.mat(x_aposteriori_k_x)

    a = np.concatenate([np.eye(3), np.zeros([3,3])], axis=1)
    b = np.concatenate([-x_aposteriori_k_x * dt, bCn], axis=1)
    PHI = np.concatenate([a, b])

    a = np.concatenate([np.eye(3) * dt, np.zeros([3,3])], axis=1)
    b = np.concatenate([np.zeros([3,3]), -x_aposteriori_k_x * dt], axis=1)
    GAMMA = np.concatenate([a,b])
    GAMMA = np.mat(GAMMA)

    a = np.concatenate([np.eye(3) * q[0], np.zeros([3,3])], axis=1)
    b = np.concatenate([np.zeros([3,3]), np.eye(3) * q[1]], axis=1)
    Q = np.concatenate([a,b])
    Q = np.mat(Q)

    P_apriori = PHI * P_aposteriori_k* PHI.T + GAMMA*Q*GAMMA.T
    R = np.eye(3) * r
    H_k = np.concatenate([np.zeros([3,3]), -g*np.eye(3)], axis=1)
    H_k = np.mat(H_k)

    K_k = (P_apriori*H_k.T) * (H_k*P_apriori*H_k.T + R).I
    x_aposteriori = x_apriori + K_k * (np.mat(a_m).T - H_k * x_apriori)
    P_aposteriori = (np.eye(6) - K_k*H_k) * P_apriori

    x_aposteriori = np.array(x_aposteriori)
    k = x_aposteriori[3:6] / np.sqrt(np.sum(x_aposteriori[3:6] ** 2))

    phi = np.math.atan2(k[1], k[2])
    theta = -np.math.asin(k[0])

    return x_aposteriori[:, 0], P_aposteriori, phi, theta



if __name__ == "__main__":
    import scipy.io as io
    import matplotlib.pyplot as plt
    data = io.loadmat('data/px4_logdata.mat')
    ax = data['ax'][0]
    ay = data['ay'][0]
    az = data['az'][0]
    gx = data['gx'][0]
    gy = data['gy'][0]
    gz = data['gz'][0]
    phi_px4 = data['phi_px4'][0]
    theta_px4 = data['theta_px4'][0]
    timestamp = data['timestamp'][0]

    n = len(ax)
    Ts = np.zeros(n)
    Ts[0] = 0.004

    for k in range(n-1):
        Ts[k+1] = (timestamp[k+1] - timestamp[k]) * 1e-6
    
    theta_am = np.zeros(n)
    phi_am = np.zeros(n)
    theta_gm = np.zeros(n)
    phi_gm = np.zeros(n)
    theta_ekf = np.zeros(n)
    phi_ekf = np.zeros(n)

    w = [0.08, 0.01]  # system noise
    v = 50  # measurement noise

    P_aposteriori = np.zeros([6, 6, n])
    P_aposteriori[:, :, 0]=np.eye(6)*100;  #P0
    x_aposteriori = np.zeros([6, n])
    x_aposteriori[:, 0] = [0, 0, 0, 0, 0, -1];  #X0

    for k in np.arange(1, n):
        # calculate Euler angles using accelerometer data
        g = np.sqrt(ax[k]*ax[k] + ay[k]*ay[k] + az[k]*az[k])
        theta_am[k] = np.math.asin(ax[k]/g)
        phi_am[k] = -np.math.asin(ay[k]/(g*np.math.cos(theta_am[k])))
        # calculate Euler angles using gyroscope data
        theta_gm[k] = theta_gm[k - 1] + gy[k]*Ts[k]
        phi_gm[k] = phi_gm[k- 1] + gx[k]*Ts[k]
        # complementary filter and EKF
        z = [gx[k], gy[k], gz[k], ax[k], ay[k], az[k]]
        x_aposteriori[:, k], P_aposteriori[:, :, k], phi_ekf[k], theta_ekf[k] = ekf(Ts[k], z, w, v, x_aposteriori[:, k - 1], P_aposteriori[:, :, k - 1])

    plt.plot(theta_ekf)
    plt.plot(theta_px4)
    plt.show()
    print('dd')