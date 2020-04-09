def getDocuments(filename):
    rows = []
    row_tuples = []
    documents = []
    with open (filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rows.append(line.split('\t'))


    for row in rows:
        if len(row) > 2:
            row_tuples.append(row)
        else:
            if len(row_tuples) > 0:
                documents.append(row_tuples)
                row_tuples = []

    documents.append(row_tuples)
    return documents

def getRelations(documents):
    document_relations = dict()
    for j , document in enumerate(documents):
        indexes_of_sentence = []
        relations_of_sentence = []

        for i,row_tuple in enumerate(document):
            #Considering only those words that are entities defined by our Base Classifier
            if row_tuple[2] != 'O':
                x = row_tuple[3].strip('\[\]')
                #If statement for incorporating the Relations other than None.
                if x[1:-1] != 'N':
                    index_of_sentence = []

                    start = i
                    end_of_sentences = row_tuple[4][1:-2].split(',')
                    index_of_sentence =  [start] + end_of_sentences

                    relations = x.split(',')

                    if j not in document_relations.keys():
                        document_relations[j] = [(relations, index_of_sentence)]
                    else:
                        document_relations[j].append((relations, index_of_sentence))
                else:
                #Incorporating the None relations

                    if row_tuple[2][0] == 'B':
                        start_index = i + 1
                        #Mapping the None relations I to I 
                        while start_index < len(document) and document[start_index][2][0] == 'I':
                            start_index= start_index + 1
                        start_index = start_index - 1
                        x = document[start_index][3].strip('\[\]')

                        if x.strip('\'') == 'N':

                            index_of_sentence = []
                            start = start_index
                            end_of_sentences = []
                            relations = []
                            for k in range(i+1,len(document)):
                                if document[k][2] != 'O' and document[k][2][0] == 'B':
                                    start_index = k + 1
                                    while start_index < len(document) and document[start_index][2][0] == 'I' :
                                        start_index= start_index + 1

                                    start_index = start_index -1 

                                    if str(start) not in document[start_index][4][1:-2].split(','): 
                                        end_of_sentences.append(start_index)
                                        relations.append('None')


                            index_of_sentence =  [start] + end_of_sentences

                            if len(relations) != 0:
    #                             print('The index of sentences are:',index_of_sentence)
                                if j not in document_relations.keys():
                                    document_relations[j] = [(relations, index_of_sentence)]
                                else:
                                    document_relations[j].append((relations, index_of_sentence))
                            
                            
    return document_relations


def getGoldTruth(document_relations,documents):
    
    gold_truths = dict()
    #dictionary of {(0,2,3):"Live_In"} 0 : is the document Number, 2,3 are the tokens in the sentence that are related by this relation which is the value of the dictionary.
    total_sentences = list()
    total_relations = []
     #Finally we annotate the sentences as per the requirement of our ML Model(BiLSTM Layer)
    for key,value in document_relations.items():
        for rel_tuple in value:
            start = rel_tuple[1][0]
            for j,relation in enumerate(rel_tuple[0]):
                end = int(rel_tuple[1][j+1])
                if start > end:
                    start ,end = end ,start

                total_relations.append(relation.strip().strip('\"').strip('\''))
                sentence = ''

                for k in range(0,len(documents[key])):

                    if int(documents[key][k][0]) == int(start) or int(documents[key][k][0]) == int(end) :
                        start_tag = '<'+ documents[key][k][2] + '>'
                        end_tag = '</'+ documents[key][k][2] + '> '
                        sentence = sentence + start_tag + documents[key][k][1] + end_tag
                    else:  
                        sentence = sentence + documents[key][k][1] + ' '

                gold_truths[(key,start,end)] = relation.strip().strip('\"').strip('\'')
                total_sentences.append(sentence)


    return (gold_truths,total_relations)
