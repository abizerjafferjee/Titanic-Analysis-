# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:14:58 2016

@author: admin
"""

import csv 
import numpy as np
import pandas as pd
import pylab as P
from sklearn.ensemble import RandomForestClassifier

#open train

openner = open('train.csv', 'r')
reader = csv.reader(openner)
data = []
for i in reader:
    if i[0] == 'PassengerId':
        continue 
    else:
        data.append(i)
data = np.array(data)

#proportion survived

number_of_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survived = number_survived / number_of_passengers

women_only_stats = data[0::,4] == 'female'
men_only_stats = data[0::,4] == 'male'
women_onboard = data[women_only_stats, 1].astype(np.int)
men_onboard = data[men_only_stats, 1].astype(np.int)

women_survived = np.sum(women_onboard) / np.size(women_onboard)
men_survived = np.sum(men_onboard) / np.size(men_onboard)

print ('proportion survived: {0}'.format(proportion_survived*100))
print ('men survived: {}'.format(men_survived*100))
print ('women survived: {}'.format(women_survived*100))

#prediction class

open_test = open('test.csv', 'r')
read_test = csv.reader(open_test)
open_predictions = open('genderbasedmodel.csv', 'w')
write_predictions = csv.writer(open_predictions)

write_predictions.writerow(['PassengerID', 'Survived'])
for row in read_test:
    if row[0] == 'PassengerId':
        continue 
    if row[3] == 'female':
        write_predictions.writerow([row[0], '1'])
    else:
        write_predictions.writerow([row[0], '0'])

open_test.close()
open_predictions.close()

#passenger class

max_fare = 40 
fare_condition = data[0::,9].astype(np.float) >= max_fare 
data[fare_condition, 9] = max_fare - 1 
bracket_size = 10
number_brackets = int(max_fare / bracket_size) 
number_classes = len(np.unique(data[0::,2]))

survival_table = np.zeros([2, number_classes, number_brackets], float)

for i in range(number_classes):
    for j in range(number_brackets):
        women_stats = data[ (data[0::,4] == "female") \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0::,9].astype(np.float) >= j*number_brackets) \
                                 & (data[0::,9].astype(np.float) < (j+1)*number_brackets), 1]

        men_stats = data[ (data[0::,4] != "female") \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0::,9].astype(np.float) >= j*number_brackets) \
                                 & (data[0::,9].astype(np.float) < (j+1)*number_brackets), 1]
        
        survival_table[0,i,j] = np.mean(women_stats.astype(np.float)) 
        survival_table[1,i,j] = np.mean(men_stats.astype(np.float))
        
survival_table[survival_table != survival_table] = 0
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

#genderclass model

# open_test = open("test.csv", "r")
# read_test = csv.reader(open_test)
# open_gender = open('genderclassmodel.csv', 'w')
# write_gender = csv.writer(open_gender)
# write_gender.writerow(['PassengerId', 'Survived'])

# for row in read_test:
#     for j in range(number_brackets):
#         try:
#             row[8] = float(row[8])
#         except:
#             bin_fare = 3 - float(row[1])
#             break
#         if row[8] > max_fare:
#             bin_fare = number_brackets - 1
#             break
#         if row[8] >= j*bracket_size and row[8] < (j+1)*brack_size:
#             bin_fare = j
#             break
#     if row[3] == 'female':
#         write_gender.writerow([row[0], "%d" % int(survival_table[0, float(row[1])-1, bin_fare])])
#     else:
#         write_gender.writerow([row[0], "%d" % \
#                    int(survival_table[1, float(row[1])-1, bin_fare])]
                              
# open_test.close()
# open_gender.close()

df = pd.read_csv('train.csv', header = 0)
for i in range(1,4):
    print (i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i)]))
    
# df['Age'].dropna().hist(bins = 8, range = (0,80), alpha = 0.5)
# P.show()
          
df['Gender'] = df['Sex'].map( {'female':0, 'male':1} ).astype(int)

median_ages = np.zeros((2,3))

for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

df['Agefill'] = df['Age']

for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'Agefill'] = median_ages[i,j]

df['Agewasnull'] = pd.isnull(df.Age).astype(int)
df['Familysize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.Agefill * df.Pclass
df.dtypes[df.dtypes.map(lambda x: x=='object')]
df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
df = df.dropna()
train_data = df.values

#test_df = pd.read_csv('test.csv', header = 0)
#test_df['Gender'] = test_df.Sex.map({'female':0, 'male':1}).astype(int)
#
#if len(test_df.Embarked[test_df.Embarked.isnull()]) > 0:
#    test_df.Embarked[test_df.Embarked.isnull()] = test_df.Embark.dropna().mode().values
#    
#Ports = list(enumerate(np.unique(test_df['Embarked'])))    # determine all values of Embarked,
#Ports_dict = { name : i for i, name in Ports }   
#    
#test_df.Embarked = test_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)
#median_age = test_df['Age'].dropna().median()
#if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
#    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age
#
#if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
#    median_fare = np.zeros(3)
#    for f in range(0,3):                                              
#        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
#    for f in range(0,3):
#        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
#
#ids = test_df['PassengerId'].values
#test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
#test_data = test_df.values


forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::], train_data[0::,0])
output = forest.predict(train_data)



                           
            




