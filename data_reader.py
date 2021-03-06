import pickle
import os
import numpy as np

def load_data():

    X = []
    Y = []
    for path in os.listdir('all_compressed_features'):

        data_point = pickle.load( open( os.path.join('all_compressed_features',path), "rb" )) 
        X.append(data_point['x'])
        Y.append(data_point['y'])

    count = 0 
    for path in os.listdir('acf2'):

        if count == 23:
            break
        count += 1
        data_point = pickle.load( open( os.path.join('acf2',path), "rb") )
        X.append(data_point['x'])
        Y.append(data_point['y'])
        

    return np.asarray(X),np.asarray(Y)


if __name__ == '__main__':

    X,Y = load_data()
    print (len(X))
    print (len(Y))
    print (Y)
    print (X[0].shape)