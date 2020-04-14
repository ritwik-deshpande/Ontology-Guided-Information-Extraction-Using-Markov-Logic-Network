
**Entity_Classifier**
<br/><br/>


**Training**
<br/>
<br/>

Please download Google Word Embeddings from 'https://github.com/bekou/multihead_joint_entity_relation_extraction/blob/master/data/CoNLL04/vecs.lc.over100freq.zip' and place them inside the data Folder before Training
<br/>
<br/>


Commands for Entity Classifier:<br/>

    cd/Entity_Classifier

For Training run the command:<br/>

    python main.py train 


**Testing**

For Testing run the command:<br/>

    python main.py test
<br/>

Results of the Entity Classifier is stored in the base_classifier_test.csv file inside the Data directory
