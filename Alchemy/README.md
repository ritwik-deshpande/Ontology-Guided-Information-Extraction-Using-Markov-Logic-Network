Use Linux (Ubuntu 16.04) as your OS for the Alchemy setup.

Download Alchemy2 (alchemy-2.tar.gz) from https://code.google.com/archive/p/alchemy-2/ . Extract the files.

In the alchemy-2/src directory, there is a makefile. Execute make command. The alchemy-2/bin/obj directory is now populated with the required object files.

If make command gives an error it is probably due to the bison version ,check your bison version. If > 3 then downgrade to 2.7 and compile the code.

To downgrade bison follow the link https://askubuntu.com/questions/444982/install-bison-2-7-in-ubuntu-14-04 .

Store the files train.db, univRules.mln, test.db and queries.txt in alchemy-2/bin.

Open a terminal and navigate to alchemy-2/bin directory. Execute the following command for training :
  
      ./learnwts -g -i univRules.mln -o output.mln -t train.db

Open the output.mln file generated. Add the following hard constraints to the weighted rules :

RFtype(s, t1, t2, OrgBased_In) ^ RFtype(s, t2, t3, Located_In) => RFtype(s, t1, t3, OrgBased_In).

RFtype(s, t1, t2, Located_In) ^ RFtype(s, t2, t3, Located_In) => RFtype(s, t1, t3, Located_In).

EFtype(s, t1, Org) ^ EFtype(s, t2, Org) => RFtype(s, t1, t2, None).

The period at the end indicates that it is a hard constraint.

Open the terminal again. Execute the following command for testing :
      
      ./infer -i output.mln -e test.db -r inference_results.results -f queries

Check the inference_results.results file for the final probabilities of classification.


To compare the final results with Base Classifier and Relation Classifier, run the results.py file present in this folder.

      python results.py

For reference :

Alchemy1 Guide:
http://alchemy.cs.washington.edu/alchemy1.html

To Run the Code:
Refer user manual http://alchemy.cs.washington.edu/user-manual/manual.html
