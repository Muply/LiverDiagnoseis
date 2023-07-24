import os
import cv2
import numpy as np

'''
缩放
'''
# 放大缩小
def BigScale(image, scale = 1.5):
    return cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
def SmallScale(image, scale = 0.5):
    return cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)


'''
翻转
'''
# 水平翻转
def Horizontal(image):
    return cv2.flip(image,1,dst=None) # 水平镜像

# 垂直翻转
def Vertical(image):
    return cv2.flip(image,0,dst=None) # 垂直镜像

# 旋转
# 可以控制角度和缩放
def Rotate(image, angle=90, scale=1):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

'''
明亮度
'''
# 变暗
def Darker(image,percetage=0.8):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(w):
        for xj in range(h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy

# 明亮
def Brighter(image, percetage=1.2):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(w):
        for xj in range(h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy

# 平移
def Move(image,x,y):
    image_info = image.shape
    height = image_info[0]
    width = image_info[1]
    mat_translation = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(image,mat_translation,(width,height))
    return dst

'''
增加噪声
'''
# 椒盐噪声
def SaltAndPepper(src, percetage=0.05):
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in  range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0,1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg

# 高斯噪声
def GaussianNoise(image, percetage=0.05):
    G_NoiseImg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_NoiseImg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_NoiseImg

def Blur(image):
    return cv2.GaussianBlur(image,(7,7),1.5)


def TestOnePic():
    test_jpg_loc = r"E:\GitHub\U-net-master\img\003\1.png"
    test_jpg = cv2.imread(test_jpg_loc)
    cv2.imshow("Show Img", test_jpg)
    #cv2.waitKey(0)

    img1 = Blur(test_jpg)
    cv2.imshow("Img1", img1)
    #cv2.waitKey(0)
    img2 = GaussianNoise(test_jpg,0.01)
    cv2.imshow("Img2", img2)
    cv2.waitKey(0)

def TestOneDir():
    root_path = "./img/013"
    save_path = root_path
    for a, b, c in os.walk(root_path):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            img_i = cv2.imread(file_i_path)

            # 放大
            img_scale1 = BigScale(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_scale1.png"), img_scale1)
            # 缩小
            img_scale2 = SmallScale(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_scale2.png"), img_scale2)
            # 水平翻转
            img_horizontal = Horizontal(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_horizontal.png"), img_horizontal)
            # 垂直翻转
            img_vertical = Vertical(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_vertical.png"), img_vertical)
            # 旋转90度 无缩放
            img_rotate = Rotate(img_i, 90, 1)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate.png"), img_rotate)
            # 变暗
            img_darker = Darker(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_darker.png"), img_darker)
            # 变亮
            img_brighter = Brighter(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_brighter.png"), img_brighter)
            # 椒盐噪声
            img_saltandpepper = SaltAndPepper(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_saltandpepper.png"), img_saltandpepper)
            # 高斯噪声
            img_gaussiannoise = GaussianNoise(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_gaussiannoise.png"), img_gaussiannoise)
            # 模糊
            img_blur = Blur(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_blur.png"), img_blur)


def TestDirs():
    root_path = "./dataset/301_CEUS_7/test"
    save_path = root_path
    dirs = os.listdir(root_path)
    for dir in dirs:
        print(dir)
        dir_path = os.path.join(root_path,dir)
        targer_path = dir_path + '_BigScale'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = BigScale(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_SmallScale'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = SmallScale(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_Horizontal'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Horizontal(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_Vertical'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Vertical(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_Rotate90'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Rotate(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_Rotate60'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Rotate(img_i,60,1)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_Darker'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Darker(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_Brighter'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Brighter(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_SaltAndPepper'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = SaltAndPepper(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_GaussianNoise'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = GaussianNoise(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
        targer_path = dir_path + '_Blur'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Blur(img_i)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)
def TestDirs2():
    root_path = "./dataset/all_data_0"
    save_path = root_path
    dirs = os.listdir(root_path)
    for i in range(1,239):
        n = ""
        if i < 10:
            n = "00" + str(i)
        elif i < 100:
            n = "0" + str(i)
        else:
            n = str(i)
        print(n)
        dir_path = os.path.join(root_path,n)
        targer_path = dir_path + '_Rotate60'
        if not os.path.exists(targer_path):
            os.makedirs(targer_path)
        for a, b, c in os.walk(dir_path):
            for file_i in c:
                file_i_path = os.path.join(a, file_i)
                # print(file_i_path)
                img_i = cv2.imread(file_i_path)
                # 放大
                img_trs = Rotate(img_i, 60, 1)
                cv2.imwrite(os.path.join(targer_path, file_i), img_trs)

if __name__ == "__main__":
    # TestOnePic()
    # TestOneDir()
    TestDirs()
    # TestDirs2()
