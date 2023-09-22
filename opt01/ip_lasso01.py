import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

def soft_thresholding(v, threshold):
    m = v.shape
    sv = np.zeros(m)
    for i in range(1, int(''.join(map(str, m)))):
        if np.abs(v[i]) <= threshold:
            sv[i] = 0
        else:
            sv[i] = v[i] - np.sign(v[i])*threshold
    return sv

# mandrill = "4.2.03.tiff"
couple = "4.1.02.tiff"

img = cv2.imread(couple)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Couple Gray", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()

height, width = img_gray.shape
noise_level = 50
noise = np.random.randint(0, noise_level, (height, width))
img_noise = img_gray + noise
img_noise[img_noise > 255] = 255
img_noise[img_noise < 0] = 0
img_noise = img_noise.astype(np.uint8)

cv2.imshow("Couple Noisy", img_noise)
cv2.waitKey()
cv2.destroyAllWindows()

Y = img_noise
opt_param = 10
Phi = np.eye(height, width)
Psi = -np.diag(np.ones(height)) + np.diag(np.ones(height - 1), 1)

N = 100
gamma = 1
X = np.zeros((height, width))
z = np.zeros(height)
v = np.zeros(height)

M = np.dot(Phi.T, Phi) + (1/gamma)*np.dot(Psi.T, Psi)

print("We Are Calculating ..........")
start = time.time()
for i in range(1, width):
    y = Y[i, :]
    w = np.dot(Phi.T,y)
    for j in range(1, N):
        x = np.linalg.pinv(M)@(w + (1/gamma)*np.dot(Psi.T, (z - v)))
        p = Psi@x + v
        z = soft_thresholding(p, gamma*opt_param)
        v = p - z
    X[i, :] = x
    print("Please Waiting ... " + str(i) + "/" + str(width))

process_time = time.time() - start
print("Finished !!!")
print("Process Time : " + str(process_time))
img_ADMM = X.astype(np.uint8)

cv2.imshow("Couple Removed Noise", img_ADMM)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("Couple_gray_true.png", img_gray)
cv2.imwrite("Couple_gray_noise.png", img_noise)
cv2.imwrite("Couple_gray_ADMM.png", img_ADMM)