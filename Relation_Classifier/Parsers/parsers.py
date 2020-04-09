def parseDocuments(filename):
    # Read the file row wise and parse each documents into tokens
    documents = []
    rows = []
    words_doc = []
    words = []
    tags = []
    with open (filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        rows.append(line.split('\t'))
        
    for row in rows:
        if len(row) > 2:
            words_doc.append((row[1],row[2]))
            words.append(row[1])
            if row[2] != 'UNK':
                tags.append(row[2])
        else:
            if len(words_doc) > 0:
                documents.append(words_doc)
                words_doc = []
    
    documents.append(words_doc)
    #print(documents)
    return documents,words,tags
