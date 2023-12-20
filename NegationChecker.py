import pickle

from textblob.classifiers import NaiveBayesClassifier
import os

# traindata = []
#
# positive_files = os.listdir("train/pos")
# negative_files = os.listdir("train/neg")
#
# directory_path = "train/pos/"
# files_to_read = 1000
#
# for i, pos_file in enumerate(os.listdir("train/pos")):
#     if i >= files_to_read:
#         break
#     file_path = os.path.join("train/pos", pos_file)
#
#     with open(file_path, encoding="utf8") as f:
#         txt = f.read().replace("<br />", " ")
#         traindata.append((txt, 'pos'))
#
#
# for i, neg_file in enumerate(os.listdir("train/neg")):
#     if i >= files_to_read:
#         break
#     file_path = os.path.join("train/neg", neg_file)
#
#     with open(file_path, encoding="utf8") as f:
#         txt = f.read().replace("<br />", " ")
#         traindata.append((txt, 'pos'))


f = open('ntlk_NegationClassifier_Blob.pickle', 'rb')
classifier = pickle.load(f)

# classifier = NaiveBayesClassifier(train)  # Pass in data as is
# When classifying text, features are extracted automatically
print('True' if classifier.classify("Netherlands is not a country") == 'pos' else 'False')

# f = open('ntlk_NegationClassifier_Blob.pickle', 'wb')
# pickle.dump(classifier, f)
# f.close()



