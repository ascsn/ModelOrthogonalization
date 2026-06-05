import numpy as np

"""
This cell will contain functions that help us to separate our data points into train, validation, and test data.
The functions include:
- separate_points_random: Separates points in list1 into two groups randomly 
  based on a given random chance.
- separate_points_distance: Separates points in list1 into two groups based on 
  their proximity to any point in list2, using a specified distance threshold.
- separate_points_distance_allSets: Separates points in list1 into three groups 
  (train, validation, test) based on their proximity to any point in list2, using 
  two specified distance thresholds.
- new_split: A function that splits a DataFrame into train, validation, and test 
  sets based on the 'Z' and 'N' columns, where we are selecting nuclei in the 
  middle for traningin, adjacent for validation, remaining for test, 
  ensuring that the training set contains no more than 50% of the data for each
"""

def separate_points_random(list1,random_chance):
    """
    Separates points in list1 into two groups randomly

    """
    train = []
    test = []

    train_list_coordinates=[]
    test_list_coordinates=[]


    for i in range(len(list1)):
        point1=list1[i]
        val=np.random.rand()
        if val<=random_chance:
            train.append(point1)
            train_list_coordinates.append(i)
        else:
            test.append(point1)
            test_list_coordinates.append(i)

    return np.array(train), np.array(test), np.array(train_list_coordinates), np.array(test_list_coordinates)

def separate_points_distance(list1, list2, distance):
    """
    Separates points in list1 into two groups based on their proximity to any point in list2.

    :param list1: List of (x, y) tuples.
    :param list2: List of (x, y) tuples.
    :param distance: The threshold distance to determine proximity.
    :return: Two lists - close_points and distant_points.
    """
    train = []
    test = []

    train_list_coordinates=[]
    test_list_coordinates=[]

    for i in range(len(list1)):
        point1=list1[i]
        close = False
        for point2 in list2:
            if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance:
                close = True
                break
        if close:
            train.append(point1)
            train_list_coordinates.append(i)
        else:
            test.append(point1)
            test_list_coordinates.append(i)

    return np.array(train), np.array(test), np.array(train_list_coordinates), np.array(test_list_coordinates)

def separate_points_distance_allSets(list1, list2, distance1, distance2):
    """
    Separates points in list1 into three groups based on their proximity to any point in list2.

    :param list1: List of (x, y) tuples.
    :param list2: List of (x, y) tuples.
    :param distance: The threshold distance to determine proximity.
    :return: Two lists - close_points and distant_points.
    """
    train = []
    validation=[]
    test = []

    train_list_coordinates=[]
    validation_list_coordinates=[]
    test_list_coordinates=[]

    for i in range(len(list1)):
        point1=list1[i]
        close = False
        for point2 in list2:
            if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance1:
                close = True
                break
        if close:
            train.append(point1)
            train_list_coordinates.append(i)
        else:
            close2=False
            for point2 in list2:
                if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance2:
                    close2 = True
                    break
            if close2==True:
                validation.append(point1)
                validation_list_coordinates.append(i)
            else:
                test.append(point1)
                test_list_coordinates.append(i)                

    return np.array(train_list_coordinates),  np.array(validation_list_coordinates), np.array(test_list_coordinates)

def new_split(df):
    # Call the array that we are going to store our train, validation, and test data
    train_set = []
    validation_set = []
    test_set= []
    train_coordinates = []
    validation_coordinates = []
    test_coordinates = []

    # create new dataframe for each Z
    for z, group in df.groupby('Z'): 
        train_idx, val_idx = [], []

        # Sort the group by 'N' but keep the original indices
        group = group.sort_values(by='N')
        
        # Extract the original indices of the sorted group
        original_indices_each_Z = group.index.to_list()
        # print(original_indices)
        
        m = len(original_indices_each_Z)

        if m == 0:
            continue

        if m <= 2:
            train_idx = original_indices_each_Z
            train_coordinates.extend(train_idx)
            continue

        # when m >= 3
        t = m // 2 # No more than 50% of the index are training
        start = (m - t) // 2 # the starting index
        train_idx = original_indices_each_Z[start: start + t]

        train_coordinates.extend(train_idx) # Add indecies are in the middle

        left = start - 1
        right = start + t

        
        if left >= 0:
            val_idx.append(original_indices_each_Z[left]) # Add the index to the validation coors
        if right < m: 
            val_idx.append(original_indices_each_Z[right]) # Ad the index to the validation coors
        
        validation_coordinates.extend(val_idx)

        chosen_idx = val_idx + train_idx
        test_idx = [i for i in original_indices_each_Z if i not in chosen_idx]

        test_coordinates.extend(test_idx)

    return train_coordinates, validation_coordinates, test_coordinates
