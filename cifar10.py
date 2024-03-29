import numpy as np
import os
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict =  pickle.load(f, encoding='latin1')
    f.close()
    return dict
def main(cifar10_data_dir):
    for i in range(1, 6):
        train_data_file = os.path.join(cifar10_data_dir, 'data_batch_' + str(i))
        #print(train_data_file)
        data = unpickle(train_data_file)
        print('unpickle done')
        for j in range(10000):
            img = np.reshape(data['data'][j], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img_name = 'train/' + str(data['labels'][j]) + '_' + str(j + (i - 1)*10000) + '.png'
            #cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)
            img = Image.fromarray(img)
            img.save(os.path.join(cifar10_data_dir, img_name))
 
    test_data_file = os.path.join(cifar10_data_dir, 'test_batch') 
    data = unpickle(test_data_file)
    print('unpickle done')
    for i in range(10000):
        img = np.reshape(data['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_name = 'test/' + str(data['labels'][i]) + '_' + str(i) + '.png'
        #cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)
        img = Image.fromarray(img)
        img.save(os.path.join(cifar10_data_dir, img_name))
 
if __name__ == "__main__":
    main('/dataset/cifar-10-batches-py')
