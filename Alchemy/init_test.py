import os
import csv
import pandas as pd
import sys

if __name__=='__main__':

	data_entity = pd.read_csv("../Data/"+sys.argv[2])

	data_relation = pd.read_csv('../Data/'+sys.argv[3])



	os.chdir(sys.argv[1])
	sentenceID = 0
	pointer_entity = 0
	pointer_relation = 0

	for j in range(1,289):
	    name = "test_"+str(j)+".db"
	    
	    with open(name,"w",newline = "") as f:
	        
	        data = data_entity.loc[data_entity['SentenceID'] == j]
	        
	        for i in range(pointer_entity,pointer_entity+len(data['Word'])):
	            string = "Etype("
	            if data['Gold Truth'][i] != "O" and data['Gold Truth'][i][2:] != "Other":
	                
	                if data['Predicted'][i] == "O":
	                    string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]-1) + "," + "Other" + ")"
	                else:
	                    string = string + str(data['SentenceID'][i]) + "," + str(data['tokenID'][i]-1) + "," + data['Predicted'][i][2:] + ")"
	                print(string,file=f)
	        pointer_entity = pointer_entity + len(data['Word'])
	        
	        data = data_relation.loc[data_relation['SentenceID'] == j]
	        
	        for i in range(pointer_relation,pointer_relation+len(data['SentenceID'])):
	            string = "Rtype(" 
	            if data['Gold Truth'][i]!= "None":
	                   
	                string = string + str(data['SentenceID'][i]) + "," + str(data['token1'][i]) + "," +str(data['token2'][i]) + "," + data['Predicted'][i] + ")"
	                print(string,file=f)
	    
	        pointer_relation = pointer_relation + len(data['SentenceID'])