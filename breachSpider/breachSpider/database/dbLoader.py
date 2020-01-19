import database as db

import sys
sys.path.append("..")
from machineLearning.MachineLearning import MachineLearning

loadfile = "../machineLearning/model.sav"

ml_trainer = MachineLearning("../docs/lemmatizedsites.txt", num_features=13400, num_estimators=100, testsplit=0.25,
                                    seed=1324640185, ngram_range=(1, 1), n_jobs=2)
ml_trainer.run(loadfile)

vect_text = ml_trainer.get_vectorized_text()
labels = ml_trainer.get_labels()
urls = ml_trainer.get_urls()
lemma_texts = ml_trainer.get_lemma_docs()

for i in range(len(labels)):
    vtext = vect_text[i].toarray()
    label = labels[i]
    url = urls[i]
    lemma_text = lemma_texts[i]

    prediction = ml_trainer.predict(vtext)[0]
    conf = ml_trainer.get_probability(vtext)[0][prediction]
    
    base_url = url.split('//')[-1].split('/')[0]
    db.threaded_db_add(BaseURL=base_url, FullURL=url, ScrapyRedirectURL=url, FullText="", LemmatizedText=lemma_text,
              Breach=prediction,conf=conf)
