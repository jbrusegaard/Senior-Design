import traceback
from MachineLearning import MachineLearning
import urllib
import lemmatizer

# loadfile = input("Input file name you would like to load: ")

loadfile = "model.sav"
# ml_trainer = MachineLearning("lemmatizedsites.txt", num_features=20000, num_estimators=100, testsplit=0.25,
#                                     seed=1572310670, ngram_range=(1, 1))
ml_trainer = MachineLearning("../docs/lemmatizedsites.txt", num_features=13400, num_estimators=100, testsplit=0.25,
                                    seed=1324640185, ngram_range=(1, 1), n_jobs=2)
ml_trainer.run(loadfile)

ml_trainer.print_stats()
ml_trainer.print_confusion_matrix()

# input("Press Enter to continue")
webpage = ""
while webpage != "EXIT":
    webpage = input("Enter a webpage: ")

    try:
        if webpage.lower() == "save":
            filename = input("Enter file name:")
            ml_trainer.save(filename)
            continue
        # elif webpage.lower() == "append":
        #     ml_trainer.add_to_model(lemmatext, labels)
        #     lemmatext.clear()
        #     labels.clear()
        #     continue

        html = urllib.request.urlopen(webpage)
        processed_text = lemmatizer.lemmatize(lemmatizer.text_from_html(html))
        # input("Press Enter to continue")
        arr = []
        arr.append(processed_text)
        vector = ml_trainer.vectorize(arr)
        # conf = 5
        # prediction = ml_trainer.predict(vector)[0]
        prediction = ml_trainer.predict(vector)[0]
        conf = ml_trainer.get_probability(vector)[0][prediction]

        # ml_trainer.add_to_model(vector, [prediction])

        # ml_trainer.add_to_model(vector, ml_trainer.predict(vector))
        # conf = 0
        if prediction != 1:
            print("The input URL was not a breach report, " + str(conf * 100) + "% confidence")
        else:
            print("The input URL was a breach report! " + str(conf * 100) + "% confidence")

    except Exception as e:
        print(e)
        traceback.print_exc()
        continue
