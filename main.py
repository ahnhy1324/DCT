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


def _8x8_inv_DCT(data):
    x = data
    z = np.zeros((B_size, B_size), dtype=float)
    y = np.zeros(B_size, dtype=float)
    ait = np.zeros(4, dtype=float)
    aixy = np.zeros(4, dtype=float)
    yy = np.zeros(B_size, dtype=float)
    outdata = np.zeros((B_size, B_size), dtype=int)  # 실제 dct값

    for ii in range(B_size):
        for jj in range(B_size):
            y[jj] = x[jj][ii]

        ait[0] = y[0]
        ait[1] = y[2]
        ait[2] = y[4]
        ait[3] = y[6]

        aixy[0] = c[15] * (ait[0] + ait[2])
        aixy[1] = c[15] * (ait[0] - ait[2])
        aixy[2] = s[7] * ait[1] - s[23] * ait[3]
        aixy[3] = c[7] * ait[1] + c[23] * ait[3]

        ait[0] = aixy[0] + aixy[3]
        ait[1] = aixy[1] + aixy[2]
        ait[2] = aixy[1] - aixy[2]
        ait[3] = aixy[0] - aixy[3]

        yy[4] = s[3] * y[1] - s[27] * y[7]
        yy[5] = s[19] * y[5] - s[11] * y[3]
        yy[6] = c[19] * y[5] + c[11] * y[3]
        yy[7] = c[3] * y[1] + c[27] * y[7]

        y[4] = yy[4] + yy[5]
        y[5] = yy[4] - yy[5]
        y[6] = -yy[6] + yy[7]
        y[7] = yy[6] + yy[7]

        yy[4] = y[4]
        yy[7] = y[7]
        yy[5] = c[15] * (-y[5] + y[6])
        yy[6] = c[15] * (y[5] + y[6])

        for jj in range(4):
            y[jj] = ait[jj] + yy[7 - jj]

        for jj in range(4):
            y[jj+4] = ait[3 - jj] - yy[jj+4]

        for jj in range(B_size):
            z[jj][ii] = y[jj]

    for ii in range(B_size):
        for jj in range(B_size):
            y[jj] = z[ii][jj]

        ait[0] = y[0]
        ait[1] = y[2]
        ait[2] = y[4]
        ait[3] = y[6]

        aixy[0] = c[15] * (ait[0] + ait[2])
        aixy[1] = c[15] * (ait[0] - ait[2])
        aixy[2] = s[7] * ait[1] - s[23] * ait[3]
        aixy[3] = c[7] * ait[1] + c[23] * ait[3]

        ait[0] = aixy[0] + aixy[3]
        ait[1] = aixy[1] + aixy[2]
        ait[2] = aixy[1] - aixy[2]
        ait[3] = aixy[0] - aixy[3]

        yy[4] = s[3] * y[1] - s[27] * y[7]
        yy[5] = s[19] * y[5] - s[11] * y[3]
        yy[6] = c[19] * y[5] + c[11] * y[3]
        yy[7] = c[3] * y[1] + c[27] * y[7]

        y[4] = yy[4] + yy[5]
        y[5] = yy[4] - yy[5]
        y[6] = -yy[6] + yy[7]
        y[7] = yy[6] + yy[7]

        yy[4] = y[4]
        yy[7] = y[7]
        yy[5] = c[15] * (-y[5] + y[6])
        yy[6] = c[15] * (y[5] + y[6])

        for jj in range(4):
            y[jj] = ait[jj] + yy[7 - jj]

        for jj in range(4):
            y[jj+4] = ait[3 - jj] - yy[jj+4]

        for jj in range(B_size):
            z[ii][jj] = y[jj] / 4.0

    for ii in range(B_size):
        for jj in range(B_size):
            outdata[ii][jj] = nint(z[ii][jj])
    return outdata


def _8X8_DCT(data):
    x = data
    z = np.zeros((B_size, B_size), dtype=float)
    y = np.zeros(B_size, dtype=float)
    ft = np.zeros(4, dtype=float)
    fxy = np.zeros(4, dtype=float)
    yy = np.zeros(B_size, dtype=float)
    outdata = np.zeros((B_size, B_size), dtype=int)#실제 dct값

    for ii in range(B_size):

        for jj in range(B_size):
            y[jj] = x[ii][jj]

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
            yy[jj+4] = y[3 - jj] - y[jj+4]

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
            y[jj] = z[jj][ii]

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
            yy[jj+4] = y[3 - jj] - y[jj+4]

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
            y[jj] = y[jj] / 4.0

        for jj in range(B_size):
            z[jj][ii] = y[jj]

    for ii in range(B_size):
        for jj in range(B_size):
            outdata[ii][jj] = nint(z[ii][jj])
    return outdata


test = np.array(0)
for i in file_name:
    image_data = image_read(i).reshape(512, 512)
    dct_image = np.zeros_like(image_data, dtype=np.int32) # 파일 입력
    print(dct_image)

    for j in range(int(image_height // B_size)):
        for k in range(int(image_height // B_size)):
            dct_image[k * B_size:(k+1) * B_size, j * B_size:(j+1) * B_size] = _8X8_DCT(image_data[k * B_size:k * B_size + 8, j * B_size:j * B_size + 8])
    plt.imshow(dct_image, cmap='gray')
    plt.show()
    for j in range(int(image_height // B_size)):
        for k in range(int(image_height // B_size)):
            dct_image[k * B_size:(k+1) * B_size, j * B_size:(j+1) * B_size] = _8x8_inv_DCT(dct_image[k * B_size:k * B_size + 8, j * B_size:j * B_size + 8])
    plt.imshow(dct_image, cmap='gray')
    plt.show()
pass
