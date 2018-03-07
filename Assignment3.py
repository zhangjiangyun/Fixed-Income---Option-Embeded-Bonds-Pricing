#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 22:59:31 2018

@author: jiangyunzhang
"""

import pandas as pd
import numpy as np
import scipy.optimize as optimize
import math

# Get data: Eurodollar and Libor Rate 
Eurodollar = pd.read_csv('Eurodollar.csv')
Libor = pd.read_csv('Libor.csv')
OSPR = pd.read_csv('Obeserved Spot Rate.csv')

# Get Current Libor Rate
LiborRate = Libor['Latest'][0]
Libor_3M_Rate = Libor['Latest'][4]
Eurodollar['Rate'] = (100 - Eurodollar['SETTLEMENT']) / 100

# Get Spot Rate Curve by Defining a Function Based on Eurodollar Futures
def SPC( EurodollarRate ):
    
    length = len(EurodollarRate)
    spotratecurve = np.zeros(length)
    spotratecurve[0] = EurodollarRate[0]
    
    for i in range(1,length):
        spotratecurve[i] = 4 * (((1 + EurodollarRate[i]/4) * (1 + spotratecurve[i-1]/4)**(i))**(1/(i+1)) - 1)
    
    return spotratecurve

# Get Curve
SpotRateCurve = SPC(Eurodollar['Rate'])


# Q1 (1)
Year = 3
Period = 4
BankAccount = np.zeros((  Year*Period + 1  , 1 ))
Interest = np.zeros((  Year*Period + 1 , 1 ))

for i in range(0,11):
    BankAccount[i] = BankAccount[i] + 90000
   
    if i%2 == 0:
        BankAccount[i] = BankAccount[i] - 1000000
    
    BankAccount[i+1] = BankAccount[i] * (1+ (Libor_3M_Rate + 2) /400)
    Interest[i+1] = - BankAccount[i] *  (Libor_3M_Rate + 2) /400

        
# Q1 (2)
BankAccount2 = np.zeros((  Year*Period + 1 , 1 ))
Interest2 = np.zeros((  Year*Period + 1, 1 ))

for i in range(0,len(BankAccount)-1):
    BankAccount2[i] = BankAccount2[i] + 90000
    
    if i%2 == 0:
        BankAccount2[i] = BankAccount2[i] - 1000000
    
    BankAccount2[i+1] = BankAccount2[i] * (1+ (Eurodollar['Rate'][i] + 0.02) /4)
    Interest2[i+1] = - BankAccount2[i] *  (Eurodollar['Rate'][i] + 0.02) /4

# Q2 1
# Get the Payment first
Payment = np.zeros(( 10, 1 ))
for i in range(0,10):
    Payment[i] = 100 * (((1 + Eurodollar['Rate'][2*i]/4) * (1 + Eurodollar['Rate'][2*i+1]/4) - 1)*2)**2 / 0.03 /2
    if i == 9:
        Payment[i] = Payment[i]+100

# Then we discounted and get price
Price = 0
DCF = np.zeros((10,1))

for i in range(0,10):
    DCF[i] =  Payment[i] / (((1 + SpotRateCurve[2*i+1]/4)*(1 + SpotRateCurve[2*i+2]/4)-1)*2 + 1)**(i/2)
    Price = DCF[i] + Price


#Q2 2
# Define function to get YTM
    
def YTM( Payment , Price , freq = 1, guess=0.05):
    freq = float(freq)
    ytm_func = lambda y: \
        sum([ Payment[t] / (1+y/freq) ** (t+1) for t in range(0,len(Payment))] ) - Price
        
    return optimize.newton(ytm_func, guess)[0]

ytm = YTM(Payment , Price[0] , freq = 2, guess=0.05)

# Calculate Duration
Duration = np.zeros((10,1))

for i in range(0,10):
    Duration[i] = Payment[i] / (1 + SpotRateCurve[2*i+2]/2)**i / Price * (i+1)/2

# Get Mac and then Mod
Mac = sum(Duration)[0]
Mod = Mac / (1+ytm/2)


# Q3
# Get Bond Price based our model is our first step

# Specify first parameters,  which are Lambda and Sigma
Parameters = np.array([0.262710504086714,0.781073791712348])

# Function to get tree and prices
def BondPrice( Parameters , Initial , TotalTime , Deltat , Coupon ):
    Lambda = Parameters[0]
    Sigma = Parameters[1]
    
    u = np.exp( Sigma * math.sqrt(Deltat))
    d = np.exp( -Sigma * math.sqrt(Deltat))
    q = (np.exp(Lambda * Deltat) - d)/(u-d)
        
    Tree = np.zeros((TotalTime,TotalTime))
    Tree[0][0] = Initial
    for i in range(0,TotalTime-1):
        for j in range(0,TotalTime-1):
            if Tree[i][j] != 0:
                Tree[i][j+1] = Tree[i][j] * u
                Tree[i+1][j+1] = Tree[i][j] * d
    
    bondprice = np.zeros(TotalTime)          
    
    for time in range(1,TotalTime + 1):
        bond = np.zeros((time,time))
        for i in range(0,time):
            for j in range(0,time):
                if j == 0: 
                    bond[i][time-j-1] = 100 + Coupon/2
                else:
                    for l in range(0,time-j):
                        bond[l][time-j-1] = (bond[l][time-j] * q + bond[l+1][time-j] * (1-q) + Coupon/2)/ (1+Tree[l][time-j-1]/2)**(Deltat*2)
        bondprice[time-1] = bond[0][0]
    return bondprice

# According to our setting
# We build a 10-period tree, which means our Delta t equal to 1.0
Initial = 0.03
TotalTime = 11
Deltat = 1

BP = BondPrice( Parameters , 0.03 , 11 , 1 , 0)
ObeserveRate = OSPR['Spot Rate']

# Calculation differences between Implied Rates and Observed Rates
def Errors( Parameters , Initial , TotalTime , Deltat , ObservedRate ):
    BP = BondPrice( Parameters , Initial , TotalTime , Deltat , 0)
    ImpliedRate = np.zeros(len(BP)-1)
    Errors = np.zeros(len(BP)-1)
    
    for i in range(1, len(BP)):
        ImpliedRate[i-1] = ((100 / BP[i])**(1/(i*2)) - 1)*2
        Errors[i-1] = (ObeserveRate[i-1] - ImpliedRate[i-1])**2
    return sum(Errors)

# Get Errors
Errors( Parameters , Initial , TotalTime , Deltat , ObeserveRate)

# Do optimization and get best parameters
result = optimize.minimize( Errors , Parameters , args=( Initial , TotalTime, Deltat , ObeserveRate ), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=None, tol=None, callback=None, options=None)
Parameters = result.x



# Q3 2
# First we can get the bond price without option
PriceNoOption = BondPrice( Parameters , 0.0275 , 16 , 0.5 , 6.25)[15]


# Price the Option
# First we should get Tree for 16 periods
def OptionPrice( Parameters , Initial , TotalTime , Deltat , Coupon , OptionTime , ExecutivePrice ):
    Lambda = Parameters[0]
    Sigma = Parameters[1]
    
    u = np.exp( Sigma * math.sqrt(Deltat))
    d = np.exp( -Sigma * math.sqrt(Deltat))
    q = (np.exp(Lambda * Deltat) - d)/(u-d)
        
    Tree = np.zeros((TotalTime,TotalTime))
    Tree[0][0] = Initial
    for i in range(0,TotalTime-1):
        for j in range(0,TotalTime-1):
            if Tree[i][j] != 0:
                Tree[i][j+1] = Tree[i][j] * u
                Tree[i+1][j+1] = Tree[i][j] * d
    
    # Calculate Bond Price
    bondprice = np.zeros(TotalTime)          
    for time in range(1,TotalTime + 1):
        bond = np.zeros((time,time))
        for i in range(0,time):
            for j in range(0,time):
                if j == 0: 
                    bond[i][time-j-1] = 100 + Coupon/2
                else:
                    for l in range(0,time-j):
                        bond[l][time-j-1] = (bond[l][time-j] * q + bond[l+1][time-j] * (1-q) + Coupon/2)/ (1+Tree[l][time-j-1]/2)**(Deltat*2)
        bondprice[time-1] = bond[0][0]
    
    # Then we try to get the Option Price
    time = OptionTime
    option = np.zeros((time,time))
    for i in range(0,time):
        for j in range(0,time):
            if j == 0: 
                option[i][time-j-1] = np.maximum( bond[i][time-j-1]-ExecutivePrice , 0 )
            else:
                for l in range(0,time-j):
                    option[l][time-j-1] = (option[l][time-j] * q + option[l+1][time-j] * (1-q) )/ (1+Tree[l][time-j-1]/2)**(Deltat*2)
    return option

# Get the price
PriceOption = OptionPrice( Parameters , 0.0275 , 16 , 0.5 , 6.25 , 6 , 100 )[0][0]
PriceWithOption = PriceNoOption - PriceOption


#Q3 3
TradingPrice = PriceWithOption * (1+0.002)
OAS = np.array([-0.00034])

def CalOAS(OAS , Initial , TotalTime , Deltat , Coupon , OptionTime , ExecutivePrice):
    Lambda = Parameters[0]
    Sigma = Parameters[1]
    
    u = np.exp( Sigma * math.sqrt(Deltat))
    d = np.exp( -Sigma * math.sqrt(Deltat))
    q = (np.exp(Lambda * Deltat) - d)/(u-d)
        
    Tree = np.zeros((TotalTime,TotalTime))
    Tree[0][0] = Initial
    for i in range(0,TotalTime-1):
        for j in range(0,TotalTime-1):
            if Tree[i][j] != 0:
                Tree[i][j+1] = Tree[i][j] * u
                Tree[i+1][j+1] = Tree[i][j] * d
    for i in range(0,TotalTime):
        for j in range(0,TotalTime):
            if Tree[i][j] != 0:
                Tree[i][j] = Tree[i][j] + OAS[0]     

    time = TotalTime 
    bond = np.zeros((time,time))
    for i in range(0,time):
        for j in range(0,time):
            if j == 0: 
                bond[i][time-j-1] = 100 + Coupon/2
            else:
                for l in range(0,time-j):
                    bond[l][time-j-1] = (bond[l][time-j] * q + bond[l+1][time-j] * (1-q) + Coupon/2)/ (1+Tree[l][time-j-1]/2)**(Deltat*2)

    time = OptionTime
    option = np.zeros((time,time))
    for i in range(0,time):
        for j in range(0,time):
            if j == 0: 
                option[i][time-j-1] = np.maximum( bond[i][time-j-1]-ExecutivePrice , 0 )
            else:
                for l in range(0,time-j):
                    option[l][time-j-1] = (option[l][time-j] * q + option[l+1][time-j] * (1-q) )/ (1+Tree[l][time-j-1]/2)**(Deltat*2)


    PriceOption = OptionPrice( Parameters , Initial , TotalTime , Deltat , Coupon , 6 , 100 )[0][0]
    PriceWithOption = PriceNoOption - PriceOption
    PriceDiff = abs(bond[0][0]-option[0][0] - PriceWithOption*1.002 )

    return PriceDiff

# Function to get diffrence in price when adding a random OAS
CalOAS(OAS , 0.0275 , 16 , 0.5 , 6.25, 6 , 100)
result = optimize.minimize( CalOAS , OAS , args=( 0.0275 , 16 , 0.5 , 6.25 , 6 , 100 ), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=None, tol=None, callback=None, options=None)

#Get the optimize OAS
OAS = result.x