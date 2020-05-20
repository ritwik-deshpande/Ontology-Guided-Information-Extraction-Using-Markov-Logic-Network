from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import sys


def getInferenceResults(filename):
    with open (filename, 'r') as f:
        lines = f.readlines()
        results_dict = dict()
        for line in lines:

            probability = float(line.split(' ')[1])
            start = 0
            while(line[start] != '('):
                start = start + 1
            start = start + 1
            end = start
            while(line[end]!=')'):
                end = end + 1

            content = line[start:end].split(',')
            # print(content,probability)
            if line[0] == 'E':
                if content[0]+'-'+content[1] in results_dict and results_dict[content[0]+'-'+content[1]][1] < probability:
                    results_dict[content[0]+'-'+content[1]] = (content[2], probability)
                elif content[0]+'-'+content[1] not in results_dict:
                    results_dict[content[0]+'-'+content[1]] = (content[2], probability)

            else:
                if content[0]+'-'+content[1]+'-'+content[2] in results_dict and results_dict[content[0]+'-'+content[1]+'-'+content[2]][1] < probability:
                    results_dict[content[0]+'-'+content[1]+'-'+content[2]] = (content[3],probability)
                elif content[0]+'-'+content[1]+'-'+content[2] not in results_dict:
                    results_dict[content[0]+'-'+content[1]+'-'+content[2]] = (content[3],probability)
        # print(results_dict)
        return results_dict

def getClassifierReport(file,label,labels):
	data = pd.read_csv(file) 

	predicted_tags = data['Predicted']
	gold_truths = data['Gold Truth']

	wo_none_predicted_tags = []
	wo_none_gold_truths = []
	if label == 'Base Classifier':
		for p_tag,g_tag in zip(predicted_tags,gold_truths):
			if g_tag !='O' and  g_tag[2:] != 'Other':
				if g_tag == "O":
					g_tag = "B-Other"
				if p_tag == "O":
					p_tag = "B-Other"
				if len(p_tag) > 2:
					wo_none_predicted_tags.append(p_tag[2:])
				else:
					wo_none_predicted_tags.append(p_tag)
				if len(g_tag)>2:
					wo_none_gold_truths.append(g_tag[2:])
				else:
					wo_none_gold_truths.append(g_tag)


	else:
		for p_tag,g_tag in zip(predicted_tags,gold_truths):
			if g_tag != 'None':
				wo_none_gold_truths.append(g_tag)
				wo_none_predicted_tags.append(p_tag)



	print('The Classification Report of ',label)
	print(classification_report(wo_none_gold_truths,wo_none_predicted_tags))
	
	return wo_none_predicted_tags,wo_none_gold_truths

def getAlchemyReport(file,label,mln_results,labels):
    THRESHOLD = 0.1
    data = pd.read_csv(file)
    actual_tags = []
    predicted_tags = []
    
   
    if label == 'Base Classifier':
        words = []
        THRESHOLD = 0.1
        for sentenceID,word,tokenID, gold_truth in zip(data['SentenceID'],data['Word'],data['tokenID'],data['Gold Truth']):
            key = str(sentenceID) + '-'+str(tokenID-1)
            if gold_truth !='O' and 'Other' not in gold_truth:
	            words.append(word)
	            if len(gold_truth) > 2:
	                actual_tags.append(gold_truth[2:])
	            else:
	                actual_tags.append(gold_truth)
	            
	            if key in mln_results.keys():
	                predicted_tags.append(mln_results[key][0])
	            else:
	                predicted_tags.append('Other')
            # if gold_truth =='O':
            #     gold_truth = 'B-Other'
            # words.append(word)
            # if len(gold_truth) > 2:
            #     actual_tags.append(gold_truth[2:])
            # if key in mln_results.keys():
            #     predicted_tags.append(mln_results[key][0])
            # else:
            #     predicted_tags.append('Other')

        
    else:
        sentences = []
        THRESHOLD = 0.005
        for sentenceID,sentence,token1,token2,gold_truth in zip(data['SentenceID'],data['Sentence'],data['token1'],data['token2'],data['Gold Truth']):
            key = str(sentenceID) + '-'+str(token1)+'-'+str(token2)
            if gold_truth != 'None':
	            sentences.append(sentence)
	            actual_tags.append(gold_truth)
	            if key in mln_results.keys() :
	                predicted_tags.append(mln_results[key][0])
	            else:
	                predicted_tags.append('None')
        

    
    # print(actual_tags)
    # print(predicted_tags)
    print('The Classification Report Alchemy trained '+label)
    print(classification_report(actual_tags,predicted_tags))
    print(labels)
    print(confusion_matrix(actual_tags,predicted_tags,labels=labels))
    if label == 'Base Classifier':
        return predicted_tags,actual_tags,words
    else:
        return predicted_tags,actual_tags,sentences

    
if __name__=='__main__':
    mln_results = getInferenceResults(sys.argv[1])
    


    entities = ['Peop','Org','O','Loc']
    relations = ['Work_For','Live_In','Kill','Located_In','None','OrgBased_In']
    p_c_tags ,a_tags = getClassifierReport("../Data/"+sys.argv[3],'Relation Classifier',relations)
    p_a_tags ,a_tags,sentences = getAlchemyReport('../Data/'+sys.argv[3],'Relation Classifier',mln_results,relations)
    
    # for p_c_tag , p_a_tag , a_tag ,sentence in zip(p_c_tags,p_a_tags,a_tags,sentences):
    # 	if p_a_tag == a_tag and p_c_tag != a_tag:
    # 		print(p_c_tag+"\t"+p_a_tag+"\t"+a_tag+"\t"+sentence)

    p_c_tags ,a_tags = getClassifierReport("../Data/"+sys.argv[2],'Base Classifier',entities)
    p_a_tags ,a_tags,words = getAlchemyReport('../Data/'+sys.argv[2],'Base Classifier',mln_results,entities)


    for p_c_tag , p_a_tag , a_tag , word in zip(p_c_tags,p_a_tags,a_tags,words):
    	if p_a_tag == a_tag and p_c_tag != a_tag:
    		print(p_c_tag+"\t"+p_a_tag+"\t"+a_tag+"\t"+word)