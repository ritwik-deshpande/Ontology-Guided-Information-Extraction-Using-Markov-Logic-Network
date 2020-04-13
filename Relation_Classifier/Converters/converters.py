def getRelations(documents):
    document_relations = dict()
    for j , document in enumerate(documents):
        indexes_of_sentence = []
        relations_of_sentence = []
        for i , row_tuple in enumerate(document):
            if row_tuple[2] != 'O':
                x = row_tuple[3].strip('\[\]')
                if x[1:-1] != 'N':
                    index_of_sentence = []
                    start = i
                    end_of_sentences = row_tuple[4][1:-2].split(',')
                    index_of_sentence = [start] + end_of_sentences
                    relations = x.split(',')
                    if j not in document_relations.keys():
                        document_relations[j] = [(relations, index_of_sentence)]
                    else:
                        document_relations[j].append((relations, index_of_sentence))
                else:
                    if row_tuple[2][0] == 'B':
                        start_index = i + 1
                        while start_index < len(document) and document[start_index][2][0] == 'I':
                            start_index = start_index + 1
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
                                    while start_index < len(document) and document[start_index][2][0] == 'I':
                                        start_index = start_index + 1
                                    start_index = start_index - 1
                                    if str(start) not in document[start_index][4][1:-2].split(','):
                                        end_of_sentences.append(start_index)
                                        relations.append('None')
                            index_of_sentence = [start] + end_of_sentences
                            if len(relations) != 0:
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
    n_relations = len(set(total_relations))
    return (gold_truths,total_relations,total_sentences)


def getIndexes(data,column,pointer):
    indexes = []
    input_words = []
    input_tags_all = []
    input_sentences = []
    input_tags_not_other =[]
    j=0;
    for i in range(pointer,pointer+len(data['Word'])):
        input_words.append(data['Word'][i])
        input_tags_all.append(data[column][i])
        if data[column][i] != 'O':
            input_tags_not_other.append((j,data[column][i]))
        j = j + 1
    flag = 0
    for i in range(0,len(input_tags_not_other)):
        for j in range(i+1,len(input_tags_not_other)):
            if input_tags_not_other[i][1][0] == 'B' and input_tags_not_other[j][1][0] == 'B':
                flag = 0			
                start1 = i + 1 
                while start1< len(input_tags_not_other) and input_tags_not_other[start1][1][0] == 'I':
                    start1 = start1 + 1 
                    flag = 1	
                start1 = start1 - 1
                start2 = j + 1
                while start2 < len(input_tags_not_other) and input_tags_not_other[start2][1][0] == 'I':
                    start2 = start2 + 1 
                    flag = 1	
                start2 = start2 - 1 
                if  start1 != start2:
                    indexes.append((input_tags_not_other[start1][0],input_tags_not_other[start2][0]))
							
    for index in indexes:
        start = index[0]
        end = index[1]
        input_sentence = ''
        for j,word in enumerate(input_words):
            if j  == start or j  == end:
                start_tag = '<'+ input_tags_all[j]+'>'
                end_tag = '</'+ input_tags_all[j]+'> '
                input_sentence = input_sentence + start_tag + word + end_tag
            else:
                input_sentence = input_sentence + word + ' '
        input_sentences.append(input_sentence)

    return input_sentences,indexes



def getGoldTruthTest(gold_truths_test,sentenceID,indexes):

    if (sentenceID,indexes[0],indexes[1]) not in gold_truths_test:
        return "None"
    return gold_truths_test[(sentenceID,indexes[0],indexes[1])]
















