import cv2
import os


# 数据预处理，把图片数据集的所有图片修剪成固定大小形状
def image_tailor(input_dir, out_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # file为root目录中的文件
            filepath = os.path.join(root, file)     # 连接两个或更多的路径名组件，filepath路径为/root/file
            try:
                image = cv2.imread(filepath)        # 根据输入路径读取照片
                dim = (227, 227)                    # 裁剪的尺寸
                resized = cv2.resize(image, dim)    # 按比例将原图缩放成227*227
                path = os.path.join(out_dir, file)  # 保存的路径和相应的文件名
                cv2.imwrite(path, resized)          # 进行保存
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey()

input_patch = '\dataset'  # 数据集的地址
out_patch = '\dataset\fix_kagglecatsanddogs_5340\\tailor'  # 图片裁剪后保存的地址
image_tailor(input_patch, out_patch)
print('reshape finished')