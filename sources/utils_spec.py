
import os
from pathlib import Path
from tqdm import tqdm
import random

import numpy as np

from sklearn.preprocessing import StandardScaler
'''
overSampling():

A) Parameters:
- touchData: numpy array of touchscreen data extracted from csv files.
- maxLength: integer which denotes the maximum length wanted for the touchData.
- touchData must have the same length as sensor data. 

B) Output:
-  Returns a new touchData array with the same length as sensorData array

C) overSampling uses two methods from random module:
- 1) random.choices(): to generate random list from a given sequence of indices
random.choices() is used if the number of indices >= length of touchData

ex: 
- missignList = random.choices([0, 1, 2], k = 20), here the given list has 3 elements and indices=20
- Length of the list=3 can be less than the number of indices=20. 

- 2) random.sample(): to generate random list from a given sequence of indices
random.sample() is used only if the number of indices < length of touchData

ex:
- missingList = random.sample(range(0,10), 10), given sequence has 10 elements, inidces can be <=10. 
'''


def overSampling(touchData, maxLength):

    missingValues = maxLength - len(touchData)
    if missingValues >= len(touchData):

        touchList = list(range(len(touchData)))
        # missingIndices : generated random indices in range: touchList with max values = missingValues
        missingIndices = random.choices(touchList, k=missingValues) 
        missingData = []
        for i in range(len(missingIndices)):
            missingData.append(touchData[missingIndices[i],:])

        missingData = np.array(missingData)
        touchArray = np.vstack([touchData, missingData])
    else:
        # if the length of the missing values is < to the length of touchData 
        # we use random.sample()
        
        missingIndices = random.sample(range(0,missingValues), missingValues)
        missingData = touchData[missingIndices,:]
        touchArray = np.vstack([touchData, missingData])
        
    return touchArray


"""
ETLHelper takes an input, ex: userPath=0 (userId between 0 and 99), and a session ex: sessionPath=0. 
ETL Helper reads out the sensor csv files with numpy.genfromtxt() function and stack horizontally the files and returns the result
"""
def ETLHelper(userID, sessionPath):

    Window_size = 200
    step = 10

    home = "D:/OptimIA/individu_projects/youcef_script/Data_1"
    # print(home)
    # Mohamed default cwd--->: "/Data_1/"
    userPath = home + '/' + str(userID) +'/'
    sessionPath = str(sessionPath) + '/'
    path = Path(userPath, sessionPath)
    os.chdir(path)

    
    acc = np.genfromtxt('Accelerometer.csv', delimiter=',')[:10000,3:6]
    gyr = np.genfromtxt('Gyroscope.csv', delimiter=',')[:10000,3:6]
    mag = np.genfromtxt('Magnetometer.csv', delimiter=',')[:10000,3:6]

    # Compute the norm for each sensor
    acc_norm = np.sqrt((acc[:,0])**2 + (acc[:,1])**2 + (acc[:,2])**2)
    gyr_norm = np.sqrt((gyr[:,0])**2 + (gyr[:,1])**2 + (gyr[:,2])**2)
    mag_norm = np.sqrt((mag[:,0])**2 + (mag[:,1])**2 + (mag[:,2])**2)
    
    # Add a dimension for each sensor: (nrow_, 1 column)
    acc_norm = np.expand_dims(acc_norm, axis=1)
    gyr_norm = np.expand_dims(gyr_norm, axis=1)
    mag_norm = np.expand_dims(mag_norm, axis=1)
    
    touchActivity = np.genfromtxt('ScrollEvent.csv', delimiter=',')[:10000,[6,7,8,9]]
    
    if len(touchActivity) < 10000:
        touchActivity = overSampling(touchData=touchActivity, maxLength=10000) 

    # Shape of touch activity: array = (10000, 4) after oversampling
    
    logs = [acc_norm, gyr_norm, mag_norm]

    os.chdir(home) # return to home directory once the data has been loaded from a user folder
    
    #np.hstack(logs)

    return  mag, touchActivity


"""
ETL() takes a user target ID with and a session list : [0, 1, 2].
Using the ETLHelper(), the ETL() function loops through the sessions and returns (or yield) the data by stacking each session vertically.
"""
def ETL(user, session):
    """
    Modifications :

    I-  sessionSet = np.array([])   --->    sessionSet = []
    
    II- sessionSet = np.vstack([s]) --->    sessionSet.append(s)
    """
    sessionSet = []
    Logs = []
    touchSet = []
    for i in tqdm(range(len(session))):
        logs, touch_logs =  ETLHelper(userID=user, sessionPath=session[i])


        #sessionSet.append(s) #append all data
        Logs.append(logs)
        touchSet.append(touch_logs)

    #sessionSet = np.vstack(sessionSet)
    Logs = np.stack(Logs)
    touchSet = np.vstack(touchSet)
    
    userLabel = np.zeros(touchSet.shape[0]) + user
    yield touchSet, userLabel, Logs


def dataGenerator(numUsers, mode = "pretraining"):

    if mode == "pretraining":
        starting_session = 0
        ending_session = 9
    elif mode == "evaluation":
        starting_session = 12
        ending_session = 24
    elif mode == "finetuning":
        starting_session = 0
        ending_session = 12
    else:
        raise Exception("Please choose the mode of training : pretraining, evaluation, finetuning.")

    x_ds = []
    touch_ds = []

    y_ds = []

    Logs = []
    y = []
    u=0


    while u < numUsers:
        print("userID {}".format(u))
        try:
            data = ETL(user=u, session=range(starting_session, ending_session))

            for touch_data, label, logs in data:
                touch_ds.append(touch_data)
                y_ds.append(label)
                Logs.append(logs)
                y.append(np.zeros(Logs[0].shape[0]) + u)

        
        except:
            print("Skip UserID {}".format(u))

        u+=1
        
    Logs = np.vstack(Logs)
    y = np.hstack(y)
    touch_ds = np.vstack(touch_ds)
    print(Logs.shape)
    touch_ds = touch_ds.reshape(Logs.shape[0], Logs.shape[1], 4)
    print(touch_ds.shape)
    print(y.shape)

    return Logs, touch_ds, y

