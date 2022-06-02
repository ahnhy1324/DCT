########################################################################################################################
#                                                   2017253019 안희영                                                   #
########################################################################################################################
import numpy as np
import math
import matplotlib.pyplot as plt

B_size = 8
image_height = 512
image_width = 512
file_name = ["lena_raw_512x512.raw", "BOAT512.raw.raw"]
pi = 3.141592653589793238
c = np.zeros(40, dtype=float)
s = np.zeros(40, dtype=float)

for i in range(40):  # 싸인 코사인 테이블
    zz = pi * (i + 1) / 64.0
    c[i] = math.cos(zz)
    s[i] = math.sin(zz)


def image_read(file):
    with open(file, "rb") as f:
        rectype = np.dtype(np.uint8)
        data = np.fromfile(f, dtype=rectype)
    return data


def image_write(name, image):
    with open(name, "wb") as f:
        f.write(bytes(np.flipud(image)))


def nint(x):
    if x < 0:
        return int(x - 0.5)
    else:
        return int(x + 0.5)


def _8X8_DCT(data):
    y = np.zeros(B_size, dtype=float)
    z = np.zeros((B_size, B_size), dtype=float)
    ft = np.zeros(4, dtype=float)
    fxy = np.zeros(4, dtype=float)
    yy = np.zeros(B_size, dtype=float)
    test = np.zeros(B_size * B_size, dtype=float)
    for ii in range(B_size):
        for jj in range(B_size):
            y[jj] = data[ii][jj]
        for jj in range(4):
            ft[jj] = y[jj] + y[7 - jj]
        fxy[0] = ft[0] + ft[3]
        fxy[1] = ft[1] + ft[2]
        fxy[2] = ft[1] - ft[2]
        fxy[3] = ft[0] - ft[3]
        ft[0] = c[15] * (fxy[0] + fxy[1])
        ft[2] = c[15] * (fxy[0] - fxy[1])
        ft[1] = s[7] * fxy[2] + c[7] * fxy[3]
        ft[3] = -s[23] * fxy[2] + c[23] * fxy[3]
        for jj in range(4):
            yy[jj] = y[7 - jj] - y[jj];
        y[4] = yy[4]
        y[7] = yy[7]
        y[5] = c[15] * (-yy[5] + yy[6])
        y[6] = c[15] * (yy[5] + yy[6])
        yy[4] = y[4] + y[5]
        yy[5] = y[4] - y[5]
        yy[6] = -y[6] + y[7]
        yy[7] = y[6] + y[7]
        y[0] = ft[0]
        y[4] = ft[2]
        y[2] = ft[1]
        y[6] = ft[3]
        y[1] = s[3] * yy[4] + c[3] * yy[7]
        y[5] = s[19] * yy[5] + c[19] * yy[6]
        y[3] = -s[11] * yy[5] + c[11] * yy[6]
        y[7] = -s[27] * yy[4] + c[27] * yy[7]
        for jj in range(B_size):
            z[ii][jj] = y[jj]
    for ii in range(B_size):
        for jj in range(B_size):
            test[jj] = nint(z[ii][jj])
    return test

def _8X8_invDCT(data):
    return 0


test = np.array(0)
for i in file_name:
    image_data = image_read(i).reshape(512, 512)  # 파일 입력
    for j in range(int(image_height / B_size)):
        for k in range(int(image_height / B_size)):
            if j == 0 and k == 0:
                test = _8X8_DCT(image_data[:8, :8])
            else:
                test = np.append(test,_8X8_DCT(image_data[k*B_size:k*B_size+8, j*B_size:j*B_size+8]))
    test = test.reshape(512, 512)
    plt.imshow(lut_to_data(stretch(test), test), cmap='gray')
    plt.show()
    # image_write('rmk_'+i, image_data)
pass
