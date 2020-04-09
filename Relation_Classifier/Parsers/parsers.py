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
       
