# IMPORTING REQUIRED LIBRARIES
import pandas as pd
import numpy as np
from math import  inf
import matplotlib.pyplot as plt

# FIRST WE IMPORT OUR SCENARIOS INTO A VARIABLE
scenarios=pd.read_excel('scenarios.xlsx',header=None)
# LET'S TAKE A LOOK
scenarios.head(10)
# FIRST THREE COLUMN OF OUR DATASET CONSIST OF OUR SCENARIOS
# SO WE NEED TO FILTER IT OUT FIRST AND PUT IT A NEW VARIABLE WE LIKE TO CALL MAIN_SCENARIOS
main_scenarios=scenarios.iloc[:,:3]
# WE ALSO NEED Probabilities OF EACH SCENARIO
probabilities=np.array(scenarios.iloc[:,5])
# NOW WE DEFINE OUR DISTANCE MATRIX
scenario_distance_matrix=np.zeros([100,100])

#FIRST WE DEFINE OUR minimum AND epsilon
minimum=0
epsilon=0.065

# (minimum < epsilon) makes sure that the min distance between scenarios does not surpass epsilon

while minimum<epsilon:
    n=scenario_distance_matrix.shape[0]      #this gives out number of rows in our distance matrix
    m=scenario_distance_matrix.shape[1]      #this gives out number of columns in our distance matrix
    for i in range(n):   
        for j in range(m): 
            if i==j:
                scenario_distance_matrix[i][j]=inf   # we set the distance of each scenario from itself to infinite
            else:
                a=np.array(main_scenarios.iloc[i])   # take scenario a
                b=np.array(main_scenarios.iloc[j])   # take scenario b
                dist=np.sqrt(np.sum((a-b)**2, axis=0))  # calculate the distance between these two
                scenario_distance_matrix[i][j]=dist*probabilities[i]  # update distance matrix
    minimum=scenario_distance_matrix.min()

    # NOW WE SEARCH TO SEE WHICH ROW AND COLUMN CONTAINS THE MINIMUMM VALUE
    # THEN WE PROCEED TO DELETE IT

    for i in range(n):
        for j in range(m):
            if scenario_distance_matrix[i][j]==minimum:
                row=i
                column=j
                break   

    #update scenarios
    # we delete the row and column which contains the min value from distance matrix 
    minimum=scenario_distance_matrix[row,column]
    probabilities[column]=probabilities[row]+probabilities[column]   # we add the deleted scenario probabilty to the one we kept
    probabilities=np.delete(probabilities,row,axis=0)                # we update our probability array for next loop
    # NOW WE UPDATE THE MAIN SCENARIOS DATAFRAME AND ALSO THE DISTANCE MATRIX
    main_scenarios=main_scenarios.drop(row).reset_index().drop(["index"], axis=1)
    scenario_distance_matrix=np.delete(scenario_distance_matrix,row,axis=0)
    scenario_distance_matrix=np.delete(scenario_distance_matrix,column,axis=1)
    # This Line of code is optional
    # here we wanted to make sure that our dataframe is reduced to 10 scenarion, no more or less
    if len(main_scenarios)==10:
        break
#let's see the new scenarios
print('reduced scenarios \n',main_scenarios)
print('\nupdated probabilities \n',probabilities)  
#############################################################################################
#############################################################################################
#############################################################################################
from docplex.mp.model import Model
FARMER=Model(name='farmer problem using reduced scenario')
#parameters
p=probabilities
wheat_yield = main_scenarios[0]
corn_yield = main_scenarios[1]
sugarbeet_yield = main_scenarios[2]
planting_cost=[150,230,260]
buying_price=[238,210]
selling_price=[170, 150, 36, 10]
#variables
x=FARMER.continuous_var_matrix(range(1,4),range(1,3),name='x',key_format='%s')
y=FARMER.continuous_var_cube(range(1,3),range(1,11),range(2,4),name='y',key_format='%s')
w=FARMER.continuous_var_cube(range(1,5),range(1,11),range(2,4),name='w',key_format='%s')
#constraints

# PLANTING CONSTRAINT FOR WHEAT

for t in range(1,3):
    FARMER.add_constraint_(x[1,t]+x[2,t]+x[3,t]<=500)

# LEAST AMOUNT REQUIRED FOR WHEAT CONSTRAINTS

for t in range(1,3):
    for s in range(1,11):
        FARMER.add_constraint_(wheat_yield[s-1]*x[1,t]+y[1,s,t+1]-w[1,s,t+1]>=200)

# LEAST AMOUNT REQUIRED FOR CORN CONSTRAINTS   
for t in range(1,3):
    for s in range(1,11):
        FARMER.add_constraint_(corn_yield[s-1]*x[2,t]+y[2,s,t+1]-w[2,s,t+1]>=240)

# SELLING AMOUNT SHOULD NOT EXCEED THE CULTIVATION         
for t in range(1,3):
    for s in range(1,11):
        FARMER.add_constraint_(sugarbeet_yield[s-1]*x[3,t]-w[3,s,t+1]-w[4,s,t+1]>=0)

# CONSTRAINT ON SELLING AMOUNT WITH HIGHEST PRICE FOR SUGARBEET
for t in range(2,4):
    for s in range(1,11):
        FARMER.add_constraint_(w[3,s,t]<=6000)

# CAN'T PLANT SUGARBEET ON THE SAME FARM FOR TWO YEARS STRAIGHT
FARMER.add_constraint_(x[3,2]<=x[1,1]+x[2,1])

# DEFINE OBJECTIVE FUNCTION
FARMER.minimize(sum(planting_cost[i]*x[i+1,t] for i in range(3) for t in range(1,3))+
sum(probabilities[s-1]*(buying_price[i-1]*y[i,s,t]-selling_price[k-1]*w[k,s,t]) for s in range(1,11)
for i in range(1,3) for t in range(2,4) for k in range(1,5)))

#PRINT solution
Solution=FARMER.solve()
Solution.display()


