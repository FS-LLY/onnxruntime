
import cv2
import numpy as np
import os
 
def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict =  pickle.load(f, encoding='latin1')
    f.close()
    return dict
 
 
def main(cifar100_data_dir):
    train_data_file = os.path.join(cifar100_data_dir, 'train') 
    print(train_data_file)
    data = unpickle(train_data_file)
    print('unpickle done')
    for i in range(50000):
        img = np.reshape(data['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_name = 'traindir/' + str(data['fine_labels'][i]) + '_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(cifar100_data_dir, img_name), img)
 
    test_data_file = os.path.join(cifar100_data_dir, 'test') 
    print(test_data_file)
    data = unpickle(test_data_file)
    print('unpickle done')
    for i in range(10000):
        img = np.reshape(data['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_name = 'testdir/' + str(data['fine_labels'][i]) + '_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(cifar100_data_dir, img_name), img)
 
 
if __name__ == "__main__":
    main('/dataset/cifar-100-python')
