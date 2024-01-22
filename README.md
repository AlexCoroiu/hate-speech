First, run data.py to extract and label text from .json files. This splits the unbalanced data into a training and a testing set.
Note: Twitter .jason files containing neutral, sexist and racist posts were used for this project. I do not have the rigths to make the .json files publicly available!

Then, run SVC.py, SVCpoly.py or SVCrbf.py to test the generalization performance of the 3 different classifiers. The models are saved in SVC.txt, SVCpoly.txt and SVCrbf.txt, respectively. 

The ML pipeline:
- processes the data into vectors using the nltk library
- runs the classifier using the scikit-learn library
- outputs the performance metrics in readable table format
  
