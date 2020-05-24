**Installing Alchemy**

Use Linux (Ubuntu 16.04) as your OS for the Alchemy setup.

Download Alchemy2 (alchemy-2.tar.gz) from https://code.google.com/archive/p/alchemy-2/ . Extract the files.

In the alchemy-2/src directory, there is a makefile. Execute make command. The alchemy-2/bin/obj directory is now populated with the required object files.

If make command gives an error it is probably due to the bison version ,check your bison version. If > 3 then downgrade to 2.7 and compile the code.

To downgrade bison follow the link https://askubuntu.com/questions/444982/install-bison-2-7-in-ubuntu-14-04 .





Copy the univRules.mln and queries.txt files from Project/Alchemy directory and place them inside alchemy-2/bin.


**Training Alchemy**

Sentence wise Training for 1153 documents

We have already provided a Directory TrainDB which contains train.db for each individual sentence. Copy them inside your alchemy-2/bin folder

If you want to generate a custom train.db using the new results that are generated from the classifers(base_classifier_train.csv and relation_classifier_train.csv) run the init_train.py file in inside your the Project/Alchemy folder. State the path of your alchemy-2/bin folder as parameter to create your custom_train.db inside that folder. Also Mention the training files as other two parameters which will be used to create train.db file (Refer the Data Folder). You can also specify no of sentences to be trained on by setting the variable NO_OF_SENTENCES inside init_train.py file. (By default set to 1153 as per our dataset)


      	python init_train.py path_of_alchemy-2/bin base_classifier_train.csv relation_classifier_train.csv




Open a terminal and navigate to alchemy-2/bin directory.

Open the terminal again. Execute the following command for training:

		for i in {1..1153}
		do
			./learnwts -g -i output_$((i-1)).mln -o output_$i.mln -t train_$i.db
		done

This will generate 1153 output.mln files. The last ouput file i.e output_1153.mln denotes the final weights of the rules after training for 1153 sentences.

Open this output_1153.mln file generated. Add the following rules as hard constraints(INF wieght) at the end of the file  :

EFtype(s,t1,Peop) ^ EFtype(s,t2,Peop) => RFtype(s,t1,t2,Kill).

EFtype(s,t1,Loc) ^ EFtype(s,t2,Loc) => RFtype(s,t1,t2,Located_In).

EFtype(s,t1,Peop) ^ EFtype(s,t2,Org) => RFtype(s,t1,t2,Work_For).

EFtype(s,t1,Org) ^ EFtype(s,t2,Loc) => RFtype(s,t1,t2,OrgBased_In).

EFtype(s,t1,Peop) ^ EFtype(s,t2,Loc) => RFtype(s,t1,t2,Live_In).





Note: Remove the weighted predicates that are present at the end of Output.mln file.

The period at the end indicates that it is a hard constraint.

You can refer the output_1153.mln file present in the Project/Alchemy Directory.


**Testing in Alchemy**

Sentence wise Inference for 288 Sentences:


We have provided the 288 generated test.db files inside TestDB folder based on our previous results. Copy them inside your alchemy-2/bin folder

If you want to generate new custom test.db files based on the new results from the classifers(base_classifier_test.csv and relation_classifier_test_base.csv), for each sentence then Run the following command to generate these files inside the Project/Alchemy folder. State the path of your alchemy-2/bin folder as parameter to create your custom_test.db files inside that folder.Also Mention the testing files as other two parameters which will be used to create test.db files(Refer the Data Folder)


      	python init_test.py path_of_alchemy-2/bin base_classifier_test.csv relation_classifier_test_base.csv


Open a terminal and navigate to alchemy-2/bin directory.

Open the terminal again. Execute the following command for testing:

		for i in {1..288}
		do
			./infer -i output_1153.mln -e test_$i.db -r infer_$i.results -f queries
		done
This will generate 288 .results file for every sentence which we combine to give a single inference.results file.
      
      	for i in {1..288}
		do
			cat infer_$i.results >> inference.results
		done


Check the inference.results file for the final probabilities of classification.
Copy the inference.results file and place it inside Project/Alchemy folder

To compare the final results with Base Classifier and Relation Classifier, run the results.py file present in this folder.

		python results.py inference.results base_classifier_test.csv relation_classifier_test_base.csv




**Assumption**

Using our original Hypothesis we are only improving the "not None" relations which were misclassified by relation Classifier in our training dataset. We are not concerned with the False positives for None relation ( because we have written the rules/hard constraints considering that we would receive Rtypes of relations whose Gold Truth is not none while Testing in the test.db files) ,hence we are not displaying the None relations in the Classification reports.


For reference :

Alchemy1 Guide:
http://alchemy.cs.washington.edu/alchemy1.html

To Run the Code:
Refer user manual http://alchemy.cs.washington.edu/user-manual/manual.html
