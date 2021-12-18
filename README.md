# Question-Answering-System---NLP-Techniques

Instructions to run :-

1. Download elasticsearch server (version 17.15.1) from the link -> https://www.elastic.co/start
2. Start the elasticsearch server by running the elasticsearch batch file inside bin folder
3. Install elasticsearch using python -> pip install elasticsearch
4. Run the requirements.txt file inside the project folder -> This will install spacy, nltk and en-core-web-sm

Python version : 3.7+

Input:
All the articles are found in the Articles folder
The list of questions is found in the questions.txt file under the root folder(QANLP folder)

Task 1 -> Run the feature-extraction.py file -> python feature_extraction.py(present in parsing folder)
Task 2 -> Run the QA-pipeline.py file -> python QA-pipeline.py(present in QANLP folder)
          Output file qa-ans.csv file will be generated under the root folder which contains the question, docId, answer sentence

