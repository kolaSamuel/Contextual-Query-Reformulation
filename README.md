# Contextual-Query-Reformulation

The files can be split into three main categories. 

The Data folder contains all the organised tables that allow fast preprocessing and feature adding. It also contains evaluation data which allow for easy calculations of precison and recall. It also contains the preprocessed sequence to sequence input and output data for all the systems tested.

The Results folder contains all the translation text, trainng and validation report logs as well as the precison, recall F1 and BLEU scores on a turn by turn level for the systems. It also contains a table that contians the macro summary of these measures. 

The codes, there are 4 scripts used throught this project they are:
1. The Open Dialog project script which contains instructoion on running the models and translating/ testing
2. The Preprocess, which was used to not only pre-process the data but to organise the data into a more useful format that would allow fast processing
3. The Results, this simply contains code used in the evaluation of they systems we tested
4. Utils, this contains useful functions that we used for both preprocessing and result generation. We also included a doc tag for all the fuctions we used as it could prove useful to other researchers/individuals performing similar tasks on text.

NOTE:
the notebooks might have some parts commented or might have some specific file names as their input and this would have to be changed to the file being used to for it to work properly and generate accurate result
