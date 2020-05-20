import os
import csv
import pandas as pd

import sys

if __name__=='__main__':
	data_entity = pd.read_csv("../Data/"+sys.argv[2]) 
	data_relation = pd.read_csv("../Data/"+sys.argv[3]) 
	os.chdir(sys.argv[1])

	NO_OF_TRAINING_SENTENCES = 1152
	with open('train_'+str(NO_OF_TRAINING_SENTENCES)+'.db', 'w', newline='') as f:
		data = data_entity
		#for i in range(0,len(data['Word'])):
		i=0
		while data['SentenceID'][i] <= NO_OF_TRAINING_SENTENCES:
		    if len(data['Predicted'][i]) > 2 :
		        string = "Etype("
		        #if data['Gold Truth'][i] != "O" and data['Gold Truth'][i][2:] != "Other":
		        string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]-1) + "," + data['Predicted'][i][2:] + ")"
		        print(string,file=f)
		    else :
		        string = "Etype("
		        #if data['Gold Truth'][i] != "O" and data['Gold Truth'][i][2:] != "Other":
		        string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]-1) + ",Other" + ")"
		        print(string,file=f)
		    if len(data['Gold Truth'][i]) > 2 :
		        string = "EFtype("
		        #if data['Gold Truth'][i] != "O" and data['Gold Truth'][i][2:] != "Other":
		        string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]-1) + "," + data['Gold Truth'][i][2:] + ")"
		        print(string,file=f)
		    else:
		        string = "EFtype("
		        #if data['Gold Truth'][i] != "O" and data['Gold Truth'][i][2:] != "Other":
		        string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]-1) + ",Other" + ")"
		        print(string,file=f)
		    i = i+1
		data = data_relation
		#for i in range(0,len(data['SentenceID'])):
		i=0
		while data['SentenceID'][i] <= NO_OF_TRAINING_SENTENCES:
		    string = "Rtype("   
		    #if data['Gold Truth'][i] != "None":
		    string = string + str(data['SentenceID'][i]) + "," + str(data['token1'][i]) + "," +str(data['token2'][i]) + "," + data['Predicted'][i] + ")"
		    print(string,file=f)
		    string = "RFtype(" 
		    #if data['Gold Truth'][i] != "None":
		    string = string + str(data['SentenceID'][i]) + "," + str(data['token1'][i]) +"," +str(data['token2'][i])+ "," + data['Gold Truth'][i] + ")"
		    print(string,file=f)
		    i = i+1
