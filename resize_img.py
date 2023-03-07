import cv2
import os

builds = ['beuk', 'chung', 'hye']

raw_path = './data/' 
token_list = os.listdir(raw_path)
print(token_list)
data_path = './resized_data/'

# Start resize -------------------
for token in token_list:
    image_path = raw_path + token + '/'
    save_path = data_path + token + '/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    data_list = os.listdir(image_path)
    print(len(data_list))

    for name in data_list:  
        im = cv2.imread(image_path + name)
        im = cv2.resize(im, (244, 244))
        cv2.imwrite(save_path + name, im)
        
    print('end ::: ' + token)