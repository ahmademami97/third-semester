from termcolor import colored
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from math import  inf
from docplex.mp.model import Model
#parameters
yields={1:[3,2.5,2,3,2.5,2,3,2.5,2] , 2:[3.6,3,2.4,3.6,3,2.4,3.6,3,2.4] , 3:[24,20,16,24,20,16,24,20,16]}
planting_cost=[150,230,260]
buying_price=[238,210]
selling_price=[170, 150, 36, 10]
# i is our counter which we can use to set how many loop we want our algorithm to Run
i=1
#define E21 and e21
#these variables are used to produce cuts in each iteration
E21=np.zeros([10,3])
e21=np.zeros(10)
E11=np.zeros([10,3])
e11=np.zeros(10)
while i<=10:
    print ('\n',colored('##############################################################################################','green')
           ,'\n',
           colored('###########################################Iteration{}#######################################'.format(i),'green')
           ,'\n',colored('#############################################################################################','green'),
           '\n')
    print(colored('\n DIR:FORWARD \n','red'))


    #FORWARD DIRECTION
    #FIRST WE SOLVE NLDS1 PROBLEM
    NLDS1=Model(name='NLDS1')
    x1=NLDS1.continuous_var_matrix(range(1,4),range(1,3),name='x',key_format='%s')
    teta1=NLDS1.continuous_var_list(range(1,2),lb=-inf,name='teta1',key_format='%s')
    NLDS1.add_constraint_(x1[1,1]+x1[2,1]+x1[3,1]<=500)
    # THE LINE BLOW CHECKS WHAT ITERATION WE'RE CURRENTLY AT
    #IF WE'RE IN THE FIRST ITERATION, THEN THE TETA==0 CONSTRAINT SHOULD BE INCLUDED IN THE MODEL
    #OTHERWISE WE DONT INCLUDE IT AND ADD THE OPTIMALITY CUTS TO OUR PROBLEM
    #FOR ADDING OPTIMALITY CUTS WE USE E21 AND e21
    # E21 * X + TETA > e
    if i==1:
        NLDS1.add_constraint_(teta1[0]==0,'g')
    if i>1:
        for k in range(i-1):
            NLDS1.add_constraint_(x1[1,1] * E11[k][0] + x1[2,1] * E11[k][1] + x1[3,1] * E11[k][2] + teta1[0] >= e11[k])
    # NOW WE DEFINE OUR OBJECTIVE FUNCTION
    NLDS1.minimize(sum(planting_cost[i-1]*x1[i,j] for i in range(1,4) for j in range(1,2))+teta1[0])
    sol_NLDS1=NLDS1.solve()
    # PRINT SOLUTION IN THE OUTPUT
    sol_NLDS1.display()
    # HERE WE DEFINE A DICTIONARY WHICH IS GOING TO CONTAIN SIMPLEX MULTIPLIERS OF EACH PROBLEM
    # WE WILL USE THIS DICTIONARY LATER IN THE MODEL TO FORMULATE OPTIMALITY CUTS
    Pi_NLDS2={}
    # NOW WE SOLVE NLDS(2,1) NLDS(2,2) AND NLDS(2,3)
    for s in range(1,4):
        NLDS2=Model(name='NLDS2{}'.format(s))
        #variables
        x2=NLDS2.continuous_var_matrix(range(1,4),range(1,3),name='x',key_format='%s')
        teta2=NLDS2.continuous_var_list(range(1,4),lb=-inf,name='teta2',key_format='%s')
        y2=NLDS2.continuous_var_cube(range(1,3),range(1,4),range(2,4),name='y',key_format='%s')
        w2=NLDS2.continuous_var_cube(range(1,5),range(1,4),range(2,4),name='w',key_format='%s')
        #constraints
        NLDS2.add_constraint_(-x2[1,2]-x2[2,2]-x2[3,2]>=-500,'a')
        NLDS2.add_constraint_(sol_NLDS1[x1[1,1]] * yields[1][s-1] + y2[1,s,2] - w2[1,s,2] >= 200,'b')
        NLDS2.add_constraint_(sol_NLDS1[x1[2,1]] * yields[2][s-1] + y2[2,s,2] - w2[2,s,2] >= 240,'c')
        NLDS2.add_constraint_(sol_NLDS1[x1[3,1]] * yields[3][s-1] - w2[3,s,2] - w2[4,s,2] >= 0,'d')
        NLDS2.add_constraint_(-w2[3,s,2]>=-6000,'e')
        # sugar beets cannot be planted two successive years on the same field 
        NLDS2.add_constraint_(sol_NLDS1[x1[1,1]] + sol_NLDS1[x1[2,1]] - x2[3,2]>=0,'f')

        # LIKE BEFORE IF WE ARE IN FIRST ITERATION WE ADD TETA=0 CONSTRAINT
        #OTHERWISE WE ADD OPT CUTS
        if i==1:
                NLDS2.add_constraint_(teta2[s-1]==0,'g')
        if i>1:
            for k in range(i-1):
                NLDS2.add_constraint_(E21[k][0] * x2[1,2] + E21[k][1] * x2[2,2] + E21[k][2] * x2[3,2] + teta2[s-1] >= e21[k])
    
        #objective function
        NLDS2.minimize(150*x2[1,2]+230*x2[2,2]+260*x2[3,2]+sum(buying_price[i-1]*y2[i,s,2] for i in range(1,3))+
                    sum(selling_price[j-1]*w2[j,s,2] for j in range(1,5))+teta2[s-1])   
        sol_NLDS2=NLDS2.solve()
        sol_NLDS2.display()
        # HERE WE CALCULATE SIMPLEX MULTIPLIERS FOR EACH CONSTRAINT AND APPEND IT TO A LIST
        Pi=[]
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('a'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('b'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('c'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('d'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('e'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('f'))[0])
        # NOW WE ADD THE RESULTING LIST TO A DICTIONARY
        # DICTIONARY KEY IS THE NAME OF THE MODEL AND WE USE THIS DICTIONARY LATER TO FORMULATE CUTS       
        Pi_NLDS2['NLDS2{}'.format(s)]=Pi
    print('Pi multipliers for NLDS2',Pi_NLDS2)

    # NOW WE FORMULATE AND SOLVE NLDS(3,1) NLDS(3,2) ...  NLDS(3,9)
    Pi_NLDS3={}
    for s in range(1,10):
        NLDS3=Model(name='NLDS3{}'.format(s))
        #variables
        teta3=NLDS3.continuous_var_list(range(1,10),lb=-inf,name='teta2',key_format='%s')
        y3=NLDS3.continuous_var_cube(range(1,3),range(1,10),range(2,4),name='y',key_format='%s')
        w3=NLDS3.continuous_var_cube(range(1,5),range(1,10),range(2,4),name='w',key_format='%s')
        #constraints
        NLDS3.add_constraint_(sol_NLDS2[x2[1,2]] * yields[1][s-1] + y3[1,s,3] - w3[1,s,2] >= 200,'a')
        NLDS3.add_constraint_(sol_NLDS2[x2[2,2]] * yields[2][s-1] + y3[2,s,3] - w3[2,s,3] >= 240,'b')
        NLDS3.add_constraint_(sol_NLDS2[x2[3,2]] * yields[3][s-1] - w3[3,s,3] - w3[4,s,3] >= 0,'c')
        NLDS3.add_constraint_(-w3[3,s,3]>=-6000,'d')
        NLDS3.add_constraint_(teta3[s-1]==0)
        #objective function
        NLDS3.minimize(sum(buying_price[i-1]*y3[i,s,3] for i in range(1,3))+
                    sum(selling_price[j-1]*w3[j,s,3] for j in range(1,5))+teta3[s-1])
        sol_NLDS3=NLDS3.solve()
        sol_NLDS3.display()
        Pi=[]
        Pi.append(NLDS3.dual_values(NLDS3.find_matching_linear_constraints('a'))[0])
        Pi.append(NLDS3.dual_values(NLDS3.find_matching_linear_constraints('b'))[0])
        Pi.append(NLDS3.dual_values(NLDS3.find_matching_linear_constraints('c'))[0])
        Pi.append(NLDS3.dual_values(NLDS3.find_matching_linear_constraints('d'))[0])
        Pi_NLDS3['NLDS3{}'.format(s)]=Pi
    print('Pi multipliers for NLDS3',Pi_NLDS3)
    print(colored('\n DIR:BACKWARD \n','red'))
    # DIR CHANGES TO BACKWARD
    # HERE WE CALCULATE E21 AND e21 
    
    #COEFFICENT OF X VARIABLES IN THE MODEL FOR THE FIRST SCENARIO (HIGH precipitation)
    T21=np.array([[3,0,0],[0,3.6,0],[0,0,24],[0,0,0]]) # COEFFICENT OF X VARIABLES IN THE MODEL FOR THE FIRST SCENARIO (HIGH precipitation)
    T22=np.array([[2.5,0,0],[0,3,0],[0,0,20],[0,0,0]]) # COEFFICENT OF X VARIABLES IN THE MODEL FOR THE SECOND SCENARIO (AVERAGE precipitation)
    T23=np.array([[2,0,0],[0,2.4,0],[0,0,16],[0,0,0]]) # COEFFICENT OF X VARIABLES IN THE MODEL FOR THE THIRD SCENARIO (LOW precipitation)
    # HERE WE EXTRACT SIMPLEX MULTIPLIER FROM THE DICTIONARY
    Pi31=np.array(Pi_NLDS3['NLDS31']).reshape(1,4) 
    Pi32=np.array(Pi_NLDS3['NLDS32']).reshape(1,4)
    Pi33=np.array(Pi_NLDS3['NLDS33']).reshape(1,4)
    # CALCULATE E21
    E21[i-1]=(1/3)*np.matmul(Pi31,T21)+(1/3)*np.matmul(Pi32,T22)+(1/3)*np.matmul(Pi33,T23)
    print(colored('E21','red'))
    print(colored(E21[i-1],'red'))
    #E22 and E23 will be the same as E21
    # DEFINE h (RIGHT HAND SIDE OF OUR MAIN CONSTRAINTS)
    h31=np.array([200,240,0,-6000]).reshape(4,1)
    h32=h31
    h33=h31
    # CALCULATE e21
    e21[i-1]=(1/3)*np.matmul(Pi31,h31)+(1/3)*np.matmul(Pi32,h32)+(1/3)*np.matmul(Pi33,h33)
    print(colored('e21','red'))
    print(colored(e21[i-1],'red'))
    # NOW WE NEED TO CHECK IF WE SHOULD ADD THE CUT OR NOT
    # FOR INSTANCE, IF e21 - E21 * X21 >=TETA2 THEN WE SHOULD ADD THE CUT TO NLDS(2,1)
    x21=np.array([sol_NLDS2[x2[1,2]] ,sol_NLDS2[x2[2,2]] ,
    sol_NLDS2[x2[3,2]]]).reshape(3,1)
    if e21[i-1]-np.matmul(E21[i-1],x21)>=sol_NLDS2[teta2[0]]:
        print(colored('condition does not hold - we add a cut to NLDS21 problem','red'))
    else:
        print(colore('no cut needed go to t-1','red'))
    # WE CONTINUE MOVING BACKWARD
    # WE ADD THE CUT TO NLDS(2,1) NLDS(2,2) NLDS(2,3) AND FIND THE SOLUTION
    for s in range(1,4):
        NLDS2=Model(name='NLDS2{}'.format(s))
        #variables
        x2=NLDS2.continuous_var_matrix(range(1,4),range(1,3),name='x',key_format='%s')
        teta2=NLDS2.continuous_var_list(range(1,4),lb=-inf,name='teta2',key_format='%s')
        y2=NLDS2.continuous_var_cube(range(1,3),range(1,4),range(2,4),name='y',key_format='%s')
        w2=NLDS2.continuous_var_cube(range(1,5),range(1,4),range(2,4),name='w',key_format='%s')
        #constraints
        NLDS2.add_constraint_(-x2[1,2]-x2[2,2]-x2[3,2]>=-500,'a')
        NLDS2.add_constraint_(sol_NLDS1[x1[1,1]] * yields[1][s-1] + y2[1,s,2] - w2[1,s,2] >= 200,'b')
        NLDS2.add_constraint_(sol_NLDS1[x1[2,1]] * yields[2][s-1] + y2[2,s,2] - w2[2,s,2] >= 240,'c')
        NLDS2.add_constraint_(sol_NLDS1[x1[3,1]] * yields[3][s-1] - w2[3,s,2] - w2[4,s,2] >= 0,'d')
        NLDS2.add_constraint_(-w2[3,s,2]>=-6000,'e')
        NLDS2.add_constraint_(sol_NLDS1[x1[1,1]] + sol_NLDS1[x1[2,1]] - x2[3,2]>=0,'f')
        # THIS FOR LOOP ADDS OPT CUTS TO THE PROBLEM
        for x in range(i):
                NLDS2.add_constraint_(E21[x][0] * x2[1,2] + E21[x][1] * x2[2,2] + E21[x][2] * x2[3,2] + teta2[s-1] >= e21[x])
        #objective function
        NLDS2.minimize(150*x2[1,2]+230*x2[2,2]+260*x2[3,2]+sum(buying_price[i-1]*y2[i,s,2] for i in range(1,3))+
                    sum(selling_price[j-1]*w2[j,s,2] for j in range(1,5))+teta2[s-1])
        sol_NLDS2=NLDS2.solve()
        sol_NLDS2.display()
        # CALCULATE SIMPLEX MULTIPLIERS
        Pi=[]
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('a'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('b'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('c'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('d'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('e'))[0])
        Pi.append(NLDS2.dual_values(NLDS2.find_matching_linear_constraints('f'))[0])
        Pi_NLDS2['NLDS2{}'.format(s)]=Pi
    #t-1=1
    # LIKE BEFORE WE CALCULATE E11 AND e11 AND USE THESE VALUES TO FORMULATE NLDS(1) CUT
    T11=np.array([[3,0,0],[0,3.6,0],[0,0,24],[0,0,0],[0,0,0],[0,0,0]])
    T12=np.array([[2.5,0,0],[0,3,0],[0,0,20],[0,0,0],[0,0,0],[0,0,0]])
    T13=np.array([[2,0,0],[0,2.4,0],[0,0,16],[0,0,0],[0,0,0],[0,0,0]])
    Pi21=np.array(Pi_NLDS2['NLDS21']).reshape(1,6)
    Pi22=np.array(Pi_NLDS2['NLDS22']).reshape(1,6)
    Pi23=np.array(Pi_NLDS2['NLDS23']).reshape(1,6)
    # CALCULATE E11
    E11[i-1]=(1/3)*np.matmul(Pi21,T11)+(1/3)*np.matmul(Pi22,T12)+(1/3)*np.matmul(Pi23,T13)
    print(colored('E11','red'))
    print(colored(E11[i-1],'red'))
    #E22 and E23 will be the same as E21
    h21=np.array([-500,200,240,0,-6000,0]).reshape(6,1)
    h22=h21
    h23=h21
    # CALCULATE e11
    e11[i-1]=(1/3)*np.matmul(Pi21,h21)+(1/3)*np.matmul(Pi22,h22)+(1/3)*np.matmul(Pi23,h23)
    print(colored('e11','red'))
    print(colored(e11[i-1],'red'))
    x11=np.array([sol_NLDS1[x1[1,1]] ,sol_NLDS1[x1[2,1]] ,
    sol_NLDS1[x1[3,1]]]).reshape(3,1)
    # CHECK IF IT'S NEEDED TO ADD THE CUT OR NOT!
    if e21[i-1]-np.matmul(E11[i-1],x11)>=sol_NLDS1.get_value_list(teta1)[0]:
        print(colored('condition does not hold - we add a cut to NLDS1 problem','red'))
    else:
        print(colored('no cut needed - optimal solution achieved!','red'))
        break

    # IF THE CONDITION ABOVE HOLDS, THEN WE ADD THE OPT CUT TO THE NLDS(1) PROBLEM
    # CUT = E11 * X + TETA1 >= e11
    NLDS1.add_constraint_(x1[1,1] * E11[i-1][0] + x1[2,1] * E11[i-1][1] + x1[3,1] * E11[i-1][2] + teta1[0] >= e11[i-1])
    NLDS1.remove_constraint('g')
    sol_NLDS1=NLDS1.solve()
    sol_NLDS1.display()
    # ADD COUNTER
    i+=1
print(colored('\n OPTIMAL SOLUTION','yellow'))
print(colored('x11 (wheat planted in first period)','green'))
print(sol_NLDS1[x1[1,1]])
print(colored('x21 (corn planted in first period','green'))
print(sol_NLDS1[x1[2,1]])
print(colored('x31 (sugarbeet planted in first period','green'))
print(sol_NLDS1[x1[3,1]])
print(colored('x12 (wheat planted in second period)','green'))
print(sol_NLDS2[x2[1,2]])
print(colored('x22 (corn planted in second period)','green'))
print(sol_NLDS2[x2[2,2]])
print(colored('x32 (sugarbeet planted in second period)','green'))
print(sol_NLDS2[x2[3,2]])

