# decisiontree.py
"""Predict Parkinson's disease based on dysphonia measurements using a decision tree."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

# ***MODIFY CODE HERE***
ROOT = '~/OneDrive\Documents\Academic Records\College\Semester 4\Machine Learning\HW 2\data'  # change to path where data is stored
THIS = os.path.dirname(os.path.realpath(__file__))  # the current directory of this file

parser = argparse.ArgumentParser(description="Use a Decision Tree model to predict Parkinson's disease.")
parser.add_argument('-xtrain', '--training_data',
                    help='path to training data file, defaults to ROOT/training_data.txt',
                    default=os.path.join(ROOT, 'training_data.txt'))
parser.add_argument('-ytrain', '--training_labels',
                    help='path to training labels file, defaults to ROOT/training_labels.txt',
                    default=os.path.join(ROOT, 'training_labels.txt'))
parser.add_argument('-xtest', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testing_data.txt',
                    default=os.path.join(ROOT, 'testing_data.txt'))
parser.add_argument('-ytest', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testing_labels.txt',
                    default=os.path.join(ROOT, 'testing_labels.txt'))
parser.add_argument('-a', '--attributes',
                    help='path to file containing attributes (features), defaults to ROOT/attributes.txt',
                    default=os.path.join(ROOT, 'attributes.txt'))
parser.add_argument('--debug', action='store_true', help='use pdb.set_trace() at end of program for debugging')
parser.add_argument('--save', action='store_true', help='save tree image to file')
parser.add_argument('--show', action='store_true', help='show tree image while running code')

def main(args):
    print("Training a Decision Tree to Predict Parkinson's Disease")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    attributes_path = os.path.expanduser(args.attributes)

    # Load data from relevant files
    # ***MODIFY CODE HERE***
    print(f"Loading training data from: {os.path.basename(training_data_path)}")
    xtrain = np.genfromtxt(training_data_path, dtype=float, delimiter= ',', comments = None)
    print(f"Loading training labels from: {os.path.basename(training_labels_path)}")
    ytrain =  np.genfromtxt(training_labels_path, dtype=int)
    print(f"Loading testing data from: {os.path.basename(testing_data_path)}")
    xtest =  np.genfromtxt(testing_data_path, dtype=float, delimiter = ',', comments = None)
    print(f"Loading testing labels from: {os.path.basename(testing_labels_path)}")
    ytest = np.genfromtxt(testing_labels_path, dtype=int)
    print(f"Loading attributes from: {os.path.basename(attributes_path)}")
    attributes = np.genfromtxt(attributes_path, dtype=str)

    print("\n=======================")
    print("TRAINING")
    print("=======================")
    # Use a DecisionTreeClassifier to learn the full tree from training data
    print("Training the entire tree...")
    # ***MODIFY CODE HERE***
    clf = DecisionTreeClassifier(criterion = "entropy")
    clf.fit(xtrain, ytrain)

    # Visualize the tree using matplotlib and plot_tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5), dpi=150)
    # ***MODIFY CODE HERE***
    plot_tree(clf, filled = True, rounded = True, feature_names= attributes, class_names = ['healthy', 'Parkinsons'])

    if args.save:
        filename = os.path.expanduser(os.path.join(THIS, 'tree.png'))
        print(f"  Saving to file: {os.path.basename(filename)}")
        plt.savefig(filename, bbox_inches='tight')
    plt.show(block=args.show)
    plt.close(fig)

    # Validating the root node of the tree by computing information gain
    print("Computing the information gain for the root node...")
    # ***MODIFY CODE HERE***
    index = clf.tree_.feature[0]  # index of the attribute that was determined to be the root node
    thold = clf.tree_.threshold[0]  # threshold on that attribute
    gain = information_gain(xtrain, ytrain, index, thold)
    print(f"  Root: {attributes[index]}<={thold:0.3f}, Gain: {gain:0.3f}")

    # Test the decision tree
    print("\n=======================")
    print("TESTING")
    print("=======================")
    # ***MODIFY CODE HERE***
    print("Predicting labels for training data...")
    ptrain = clf.predict(xtrain)
    print("Predicting labels for testing data...")
    ptest = clf.predict(xtest)

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compare training and test accuracy
    # ***MODIFY CODE HERE***
    matching_train = np.sum(ptrain == ytrain)
    matching_test = np.sum(ptest == ytest)
    accuracy_train = matching_train / (len(ptrain))+ 0.0
    accuracy_test = matching_test / (len(ptest))+ 0.0
    print(f"Training Accuracy: {matching_train}/{len(ptrain)} ({(accuracy_train * 100):.2f}%)")
    print(f"Testing Accuracy: {matching_test}/{len(ptest)} ({(accuracy_test * 100):.2f}%)")

    # Show the confusion matrix for test data
    # ***MODIFY CODE HERE***
    print("Confusion matrix:")
    cm = confusion_matrix(ytest,ptest)
    print(str(cm).replace(" [", '').replace('[', '').replace(']', ''))
    # Debug (if requested)
    if args.debug:
        pdb.set_trace()

# Compute Information Gain
def information_gain(x, y, index, thold):
    """Compute the information gain on y for a continuous feature in x (using index) by applying a threshold (thold).

    NOTE: The threshold should be applied as 'less than or equal to' (<=)"""
    
    # ***MODIFY CODE HERE***
    gain = 0 # Information Gain
    entropy_value = entropy(y) # Entropy of Remaining Samples
    conditional_entropy_value = conditional_entropy(x, y, index, thold) # Conditional Entropy of all Values for a Feature
    gain = entropy_value - conditional_entropy_value # Subtract Conditional Entropy from Entropy of the Class (Remaining Samples)
    
    # Return Information Gain
    return gain

# Calculate Entropy for Remaining Samples
def entropy(ytrain):
    healthy = 0 # Number of Healthy Individuals
    park = 0 # Number of Individuals with Parkinsons

    # For Samples in Dataset
    for i in range(len(ytrain)):
        # If Individual is Healthy
        if ytrain[i] == 0:
            healthy += 1 # Increment Number of Healthy Individuals
        # If Individual is Diagnosed
        else:
            park += 1 # Increment Number of Diagnosed Individuals

    # Calculate Entropy of Classes
    entropy_value = -(healthy / len(ytrain))*(np.log2(healthy/len(ytrain))) -(park / len(ytrain))*(np.log2(park/len(ytrain)))

    # Return Entropy
    return entropy_value

# Calculate Conditional Entropy Classes Given a Feature
def conditional_entropy(xtrain, ytrain, index, thold):
    count_less = 0 # Number of Samples Less Than or Equal to Threshold
    count_gtr = 0 # Number of Samples Greater Than Threshold

    # For Sample in X-Train
    for element in xtrain:
        if element[index] <= thold: # If Sample <= Threshold
            count_less += 1 # Increment Number of Samples <=
        else: # If Sample > Threshold
            count_gtr += 1 # Increment Number of Samples >

    # Call Function to Calculate Specific Entropies of Feature Threshold
    spec_entropy_value = spec_entropy(xtrain, ytrain, index, thold)

    # Calculate Conditional Entropy
    conditional_entropy_value = ((count_less / len(xtrain))*(spec_entropy_value[0]) + (count_gtr / len(xtrain))*(spec_entropy_value[1]))

    # Return Conditional Entropy
    return conditional_entropy_value
    
# Calculate the Specific Conditonal Entropy of a Feature
def spec_entropy(xtrain, ytrain, index, thold):
    spec_entrop_value = [] # Placeholder for Specific Entropy
    count_less = 0 # Number of Samples Less Than or Equal to Threshold
    count_less_park = 0 # Number of Samples Less Than or Equal to Threshold that have Parkinsons
    count_less_healthy = 0 # Number of Samples Less Than or Equal to Threshold that are Healthy
    count_gtr_park = 0 # Number of Samples Greater than Threshold that have Parkinsons
    count_gtr_healthy = 0 # Number of Samples Greater than Threshold that are Healthy
    count_gtr = 0 # Number of Samples Greater than Threshold

    # For Sample in X-Train
    for i in range(len(xtrain)):
        if xtrain[i][index] <= thold: # If Feature Value of Sample is Less Than or Equal to Threshold
            count_less += 1 # Increment Number of Samples Less than or Equal to Threshold
            if ytrain[i] == 1: # If Sample has Parkinsons
                count_less_park += 1 # Increment Number of Individuals with Parkinsons
            else: # If Sample is Healthy
                count_less_healthy += 1 # Increment Number of Healthy Individuals
        else: # If Feature Value of Sample is Greater Than Threshold
            count_gtr += 1 # Increment Number of Samples Greater Than Threshold
            if ytrain[i] == 1: # If Sample has Parkinsons
                count_gtr_park += 1 # Increment Number of Individuals with Parkinsons
            else: # If Sample is Healthy
                count_gtr_healthy += 1 # Increment Number of Healthy Individuals

    # Calculate Specific Entropy Values           
    spec_entrop_value.append(-(count_less_park / count_less)*(np.log2(count_less_park / count_less)) -(count_less_healthy / count_less)*(np.log2(count_less_healthy / count_less)))
    spec_entrop_value.append(-(count_gtr_park / count_gtr)*(np.log2(count_gtr_park / count_gtr)) -(count_gtr_healthy / count_gtr)*(np.log2(count_gtr_healthy / count_gtr)))
    
    # Return Specific Entropy Values
    return spec_entrop_value

if __name__ == '__main__':
    main(parser.parse_args())