
**Entity_Classifier**

Files for training and testing should be present inside the Folder of Entity Classifier.
Make the changes inside main.py. Currently train.txt and test.txt are being used.

**Training**


Please download Google Word Embeddings from 'https://github.com/bekou/multihead_joint_entity_relation_extraction/blob/master/data/CoNLL04/vecs.lc.over100freq.zip' and place them inside the Entity_Classifier Directory before Training



Commands for Entity Classifier:
cd/Entity_Classifier

For Training run the command:
python main.py train 

For Testing run the command:
python main.py test
