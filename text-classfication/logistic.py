from skimage import io
from matplotlib import  pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import cv2

#get labels from txt file
def txt_process(filename):
    label = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        new_line = line.rstrip('\n').split("\t") #split by space and del \n
        label.append(int(int(new_line[1]) * 0.5 + 0.5)) #if label -1 -> 0, 1 -> 1
    print(filename + " processing finished")
    return label


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# cost function with regularization
def cost(w, b, X, y, factor = 5):
 
    m = X.shape[0]
    z = np.matmul(X, w) + b
    hx = sigmoid(z)

    L = (-1 / m) * ( np.sum( y * np.log(hx) + (1. - y) * np.log(1. - hx) ) )

    L += (factor / (2 * m)) * np.matmul(w,w)
    
    return L

# gradient descent with regularization
def gradient_descent(w, b, X, y, lr=0.5, factor=5, iterations=3000):

    losses = []
    size = X.shape[0]

    print("Initial cost: {}".format( cost(w, b, X, y) ))

    for i in range(iterations):

        z = np.matmul(X, w) + b
        hx = sigmoid(z)

        dw = (1 / size) * np.matmul(X.T, hx - y)
        db = (1 / size) * np.sum(hx - y)

        factor = 1 - ((lr * factor) / size)
        
        w = w * factor - lr * dw
        b = b - lr * db

        if i % 100 == 0:
            if i != 0:
                loss = cost(w, b, X, y)
                losses.append(loss)
                print("Iteration {} cost :{}".format(i, loss))
    
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('100 iterations')
    plt.title("Logistic loss(sobel filter. lr = 0.5.)")
    plt.show()
    print("Final cost: {}".format( cost(w, b, X, y) ))

    return w, b

def accuracy(w, b, X, y):

    size = X.shape[0]

    z = np.matmul(X, w) + b
    hx = sigmoid(z)

    pred = np.round(hx)

    correct_pred = (pred == y)

    total = np.sum(correct_pred)

    return (total * 100) / size



def main():

    #load data
    all_image = np.array(io.imread_collection('image_data/*.png'))

    # #feature extraction sobel filter    
    # for i in range(len(all_image)):
    #     all_image[i] = cv2.Sobel(src=all_image[i], ddepth=cv2.CV_8U, dx=1, dy=0, ksize=5) 
    # #     all_image[i] = cv2.Canny(image=all_image[i], threshold1=100, threshold2=200)

    a = plt.imshow(cv2.Sobel(src=all_image[7], ddepth=cv2.CV_8U, dx=1, dy=0, ksize=5))
    plt.show() 
    plt.savefig(a)
    return
    #train_08000_train_47999 len 40000
    train_image = all_image[8000:48000]/255 #normalize
    
    return
    train_label = np.array(txt_process('train.txt'))

    #val 48000- 57999 len 10000
    val_image = all_image[48000:58000]/255 #normalize
    val_label = np.array(txt_process('val.txt'))

    #After reshaping : (40000,28x28) = (40000,784)
    train_image = np.reshape(train_image, (len(train_image),-1)).astype(np.float64)
    val_image = np.reshape(val_image, (len(val_image),-1)).astype(np.float64)
    print("finish data process\nstart training")



    # Initializing parameters for the model
    w = np.zeros(train_image.shape[1], dtype=np.float64)
    b = 0.0

    # Performing gradient descent
    w,b = gradient_descent(w, b, train_image, train_label)

    # print(w, b)

    #print the accuracy
    print("Train accuracy: {}".format(accuracy(w, b, train_image, train_label)))
    print("Test accuracy: {}".format(accuracy(w, b, val_image, val_label)))

    

if __name__ == "__main__":
    main()