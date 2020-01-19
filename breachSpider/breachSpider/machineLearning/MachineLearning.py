from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
import os
import pandas as pd
from tabulate import tabulate


class MachineLearning:
    def __init__(self, file_name, num_features=13400, num_estimators=100, testsplit=0.25, seed=0, ngram_range=(1, 1),
                 n_jobs=-1, verbose=False):
        self.file_name = file_name
        self.labels = list()
        self.lemmatized_documents = []
        self.urls = []
        self.num_features = num_features
        self.num_estimators = num_estimators
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.useBinaryLabels = True  # Makes it so that only 0,1s used for labeling rather than 0,1,2
        self.score = 0
        self.seed = seed
        self.testsplit = testsplit
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.n_jobs = n_jobs
        self.vectorized_text = None
        self.verbose = verbose

    # Runs the model by loading a file if one is specified by default it will generate a new model
    # This needs to be called in order to fully construct and use the model
    def run(self, loadfile=None):
        # Parse out the urls, labels, and lemmatized text from the training data
        self.parse_file()
        self.labels = numpy.asarray(self.labels)
        # Vectorize the lemmatized text
        self.vectorized_text = self.vectorize()
        # If using default seed create a new seed for splitting the training and testing set
        # used for model recreation when loading purposes
        if self.seed == 0:
            random_data = os.urandom(4)
            self.seed = int.from_bytes(random_data, byteorder="big")

        if self.verbose:
            print("Training/Test Data Split Seed = %s" % self.seed)

        # Splits data into training and testing sets, X being the text, y being the labels
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_text, self.labels,
                                                            test_size=self.testsplit,
                                                            random_state=self.seed)
        # Store this in the model object for later easy access in other functions
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # If a file is not being loaded then create the model and store predictions and accuracy
        # otherwise load a model
        if loadfile is None:
            self.model = self.train_model(X_train, y_train)
            self.y_pred = self.model.predict(X_test)
            self.score = self.model.score(self.X_test.toarray(), self.y_test)
        else:
            self.load(loadfile)

    # Puts lemmatized text into vectorizer and returns list of vectors for text, return type is sparse
    # matrix from numpy
    def vectorize(self, text=None):
        if text is None:
            features = self.vectorizer.fit_transform(self.lemmatized_documents)
        else:
            features = self.vectorizer.transform(text)
        return features

    # Parses the lemmatized text file into the respective arrays/lists
    # Text goes to the lemmatized_documents list, labels go to labels list
    # and urls get stored as well for display purposes
    def parse_file(self):
        f = open(self.file_name, 'r')
        lines = f.readlines()
        f.close()
        numZeros = 0
        numOnes = 0
        for line in lines:
            line_delim = line.split("$#delimeter#$")
            self.urls.append(line_delim[0])
            self.lemmatized_documents.append(line_delim[1])
            if self.useBinaryLabels:  # Used to use 3 labels for non br, br and vuln reports found this to be worse than
                                      # using just 2 labels this is turned on in the constructor
                if int(line_delim[2]) != 1:
                    self.labels.append(0)
                    numZeros += 1
                else:
                    self.labels.append(1)
                    numOnes += 1
            else:
                self.labels.append(int(line_delim[2]))
        if self.verbose:
            print("Data points: " + str(numOnes + numZeros))
            print("    Breach reports: " + str(numOnes))
            print("    Non-Breach reports: " + str(numZeros))

    # Creates and returns a model with a certain type of machine learning model
    def train_model(self, training_features, labels):

        self.model = RandomForestClassifier(n_estimators=self.num_estimators,
                                            max_features=self.num_features, n_jobs=self.n_jobs, warm_start=True)
        self.model.fit(training_features.toarray(), labels)
        return self.model

    # Returns a prediction from the model of a given vector
    # Return type is an array of prediction values of 0 for non breach report and 1 for breach report
    def predict(self, vector):
        return self.model.predict(vector)

    # Saves model to filename, Creates a text file to save the seed and stats about the model for loading it
    # Also saves binary of model object to be reloaded into the self.model variable
    def save(self, filename):
        pickle.dump(self.model, open((filename + ".sav"), 'wb'))
        file = open((filename + ".txt"), 'w')
        file.write("Data Split Seed = " + str(self.seed))
        file.write("\n")
        file.write("Num estimators: " + str(self.num_estimators) + "\nNum features: " + str(self.num_features))
        file.write("\n")
        file.write("Accuracy Score: {0:.2f}".format(self.score * 100) + "%")
        file.write("\n")
        file.write("Matthews Correlation Coefficient: " + str(self.get_matthews_corrcoef()))
        file.write("\n")
        file.write(str(metric.confusion_matrix(self.y_test, self.y_pred)))
        file.write("\n")
        file.write("Report: ")
        file.write("\n")
        file.write(classification_report(self.y_test, self.y_pred))
        file.write("\n")
        file.write("Test Split " + str(self.testsplit))
        file.write("\n")
        file.close()

    # Loads filename into the self.model object then recreates prediction matrix and other variables needed to use
    # other functions
    def load(self, filename):
        self.model = pickle.load(open(filename, 'rb'))
        self.y_pred = self.model.predict(self.X_test)
        self.score = self.model.score(self.X_test.toarray(), self.y_test)
        self.num_estimators = self.model.n_estimators
        if self.verbose:
            print("Params: " + str(self.model.get_params()))

    # Prints the confusion matrix to the terminal this matrix provides information about False-Positives,
    # True-Positives, etc.
    def print_confusion_matrix(self):
        print(metric.confusion_matrix(self.y_test, self.y_pred))

    # Prints misc stats about the model such as the split value, Number of estimators, accuracy score, etc.
    def print_stats(self):
        print("Test split = " + str(self.testsplit))
        print("Num estimators: " + str(self.num_estimators) + "\nNum features: " + str(self.num_features))

        print("Accuracy Score: {0:.2f}".format(self.score * 100), "%", sep='')
        print("Matthews Correlation Coefficient: " + str(self.get_matthews_corrcoef()))
        print("Report: ")
        print(classification_report(self.y_test, self.y_pred))
        self.print_n_most_used_words(n=25)
        print()
        feature_importances = pd.DataFrame(self.model.feature_importances_,
                                           index=self.vectorizer.get_feature_names(),
                                           columns=['importance']).sort_values('importance', ascending=False)
        print(tabulate(feature_importances[:25], headers=['Word', 'Importance']))
        print()

    # Returns accuracy score of the model
    def get_score(self):
        return self.score

    # Returns precision score of the model for a given label
    # unless the model is changed it will be a 0 or a 1
    def get_precision_score(self, label):
        return precision_score(self.y_test, self.y_pred, pos_label=label)

    # Returns seed of the model
    def get_seed(self):
        return self.seed

    # Returns f1 score of the model for a given label
    # unless the model is changed it will be a 0 or a 1
    def get_f1_score(self, label):
        return f1_score(y_true=self.y_test, y_pred=self.y_pred, pos_label=label)

    # Returns recall score of the model for a given label
    # unless the model is changed it will be a 0 or a 1
    def get_recall_score(self, label):
        return recall_score(y_true=self.y_test, y_pred=self.y_pred, pos_label=label)

    # Returns Matthews correlation coefficient, gives more of a "weighted accuracy" for the model, takes into account
    # the label weight like if there are more of one type of data label than another
    def get_matthews_corrcoef(self):
        return matthews_corrcoef(y_true=self.y_test, y_pred=self.y_pred)

    # Returns prediction probability of a given vector, will return array of probabilities if there are more than one
    # thing being tested at the same time
    def get_probability(self, vector):
        return self.model.predict_proba(vector)

    # Returns numpy array of vectorized text (will be in the form of numpy.sparse array of floats)
    def get_vectorized_text(self):
        return self.vectorized_text

    # Returns array of labels
    def get_labels(self):
        return self.labels

    # Returns array of urls
    def get_urls(self):
        return self.urls

    # Returns list of lemmatized "documents"
    def get_lemma_docs(self):
        return self.lemmatized_documents

    # Prints the n most recurring words in the vectorizer dictionary
    def print_n_most_used_words(self, n=20):
        freqs = zip(self.vectorizer.get_feature_names(), self.vectorized_text.sum(axis=0).tolist()[0])
        # sort from largest to smallest
        top_n = sorted(freqs, key=lambda x: -x[1])
        count = 1
        print("Top " + str(n) + " Vectorized words:")
        for top in top_n[:n]:
            print(str(count) + ".) " + str(top))
            count += 1
