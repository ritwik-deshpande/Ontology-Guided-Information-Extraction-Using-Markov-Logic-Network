import os
import csv
import pandas as pd

if __name__=='__main__':
	with open('train.db', 'w', newline='') as f:
	    data = pd.read_csv("../Data/base_classifier.csv") 
	    for i in range(0,len(data['Word'])):
	        string = "Etype("
	        if data['Predicted'][i] != "O" and data['Predicted'][i][2:] != "Other":
	            string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]) + "," + data['Predicted'][i][2:] + ")"
	            print(string,file=f)
	        string = "EFtype("
	        if data['Gold Truth'][i] != "O" and data['Gold Truth'][i][2:] != "Other":
	            string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]) + "," + data['Gold Truth'][i][2:] + ")"
	            print(string,file=f)
	    data = pd.read_csv("../Data/relation_classifier.csv") 
	    for i in range(0,len(data['SentenceID'])):
	        string = "Rtype("    
	        string = string + str(data['SentenceID'][i]) + "," + str(data['token1'][i]) + "," +str(data['token2'][i]) + "," + data['Predicted'][i] + ")"
	        print(string,file=f)
	        string = "RFtype("    
	        string = string + str(data['SentenceID'][i]) + "," + str(data['token1'][i]) +"," +str(data['token2'][i])+ "," + data['Gold Truth'][i] + ")"
	        print(string,file=f)
