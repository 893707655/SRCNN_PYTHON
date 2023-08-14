import cv2
import os
def countFile(dir): #函数作用：计算文件夹中有多少张图片，返回图片数量。参数：dir目标文件夹地址，
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp


row=400 #定义修改后图片大小
col=400
# 函数作用：读取存放高分辨率图像的目录文件夹，并将高分辨率图像重新命名输出到输出目录当中
# 并以1，2，3。。。。的顺序进行命名。
# 返回值：重新命名存放高分辨率图像的路径
# 参数：input_dir:原存放高分辨率图像文件夹路径、output_dir高分辨率图像修改命名后的输出目录。
def renameHighresolutionPictures(input_dir):
    index=1  # 文件名初始化
    output_dir = os.path.join(input_dir, '..', 'highResolution')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 判断输出路径是否存在，不存在则建立
    for file in os.listdir(input_dir):   # 读取存放高分辨率图像的目录文件夹，并将高分辨率图像重新命名输出到输出目录当中
        original_image = cv2.imread(input_dir+'/'+file)  # numpy.ndarray
        original_image=cv2.resize(original_image,(row,col))
        cv2.imwrite(output_dir + '/' + str(index) + ".png", original_image)
        index = index+1
    return output_dir

# 函数作用：将修改命名后的高分辨率图像进行下采样，存放到输出目录
# 参数：filenum文件夹中图片数量、n为下采样倍数值为2或者4、input_dir为修改命名后
# 高分辨率图像存放地址、output_dir为下采样图片存放地址。
def createSamplesPictures(filenum,n,input_dir):
    index=1
    num = 1
    output_dir=os.path.join(input_dir,'..','lowResolution')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 创建低分辨率图像存储路径

    for i in range(1, filenum + 1):
    ########################################################
    # 1.读取原始图片
        original_image = cv2.imread(input_dir + '/'+str(i) + ".png")
    # 2.下采样后进行上采样填充，获得低分辨率图像
        if n == 2:
            img_1 = cv2.pyrDown(original_image)
            img_1 = cv2.pyrUp(img_1)
        if n == 4:
            img_1 = cv2.pyrDown(original_image)
            img_1 = cv2.pyrDown(img_1)
            img_1 = cv2.pyrUp(img_1)
            img_1 = cv2.pyrUp(img_1)
    # 3.将下采样图片保存到指定路径当中
        cv2.imwrite(output_dir + '/'+ str(index) + ".png", img_1)
        print("正在为第" + str(num) + "图片采样......")
        num = num + 1
        index = index + 1

if __name__ == '__main__':
    inputDir = "C:/Users/ROG/Desktop/textimage/high" # 原始高分辨率图像存放地址
    filenum = countFile(inputDir)  # 返回的是图片的张数
    print(filenum)
    outputDir = renameHighresolutionPictures(inputDir) # 将图片重新命名输出到inputDir
    createSamplesPictures(filenum,4,outputDir)         # 输出下采样后进行填充的低分辨率图像