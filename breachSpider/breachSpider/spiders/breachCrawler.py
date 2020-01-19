import scrapy
import pandas
from breachSpider.database import database as db
from breachSpider.machineLearning import lemmatizer
from breachSpider.machineLearning.MachineLearning import MachineLearning
import numpy
import random

class ItsyBitsySpider(scrapy.Spider):
    name = 'breachCrawler'

    def __init__(self, *args, **kwargs):
        super(ItsyBitsySpider,self).__init__(*args, **kwargs)
        print("HI THIS IS START URL!!!:")
        print(kwargs.get('start_url'))
        self.start_urls = [str(kwargs.get('start_url'))]
    seedOnly = False
    # allowed_domains = ['']
    colNames = ['domain', 'number']
    data = pandas.read_csv('breachSpider/docs/history.csv', names=colNames)
    seed_urls = data.domain.tolist()
    # start_urls = ['https://itblogr.com/list-of-data-breaches-and-cyber-attacks-in-may-2019-1-39-billion-records-leaked/']
    labels = numpy.asarray(data.number.tolist())
    file_name = "breachSpider/docs/lemmatizedsites2.txt"  # CHANGE THIS
    return_name = "log.csv"
    file = open(file_name, "a")
    returnfile = open(return_name, "a")
    log_file = open("brokenlinks.txt", "a")
    BLOCKED_DOMAINS = open('breachSpider/docs/blockeddomains.txt').read().splitlines()
    loadfile = "breachSpider/machineLearning/model.sav"
    ml_trainer = MachineLearning("breachSpider/docs/lemmatizedsites.txt", num_features=13400,
                                 num_estimators=100, testsplit=0.25, seed=1324640185, ngram_range=(1, 1), n_jobs=2)
    ml_trainer.run(loadfile)
    if ml_trainer is None:
        print("ERROR CREATING ML TRAINER")
        exit()
    else:
        print("Created MODEL")

    # Parsing function for scrapy
    def parse(self, response):

        bodyText = lemmatizer.text_from_html(response.body)
        lemmatized_text = lemmatizer.lemmatize(bodyText)
        # list of lemmatized documents
        if self.seedOnly:
            try:
                index = self.start_urls.index(response.request.url)
                self.file.write(response.request.url)
                self.file.write("$#delimeter#$")
                self.file.write(lemmatized_text)
                self.file.write("$#delimeter#$")
                self.file.write(str(self.labels[index]))
                self.file.write("\n")
            except Exception as e:
                print("EXCEPTION WHEN LEMMATIZING: " + str(e))
                self.log_file.write(response.request.url)
                self.log_file.write("\n")

        try:
            arr = list()
            arr.append(lemmatized_text)
            vector = self.ml_trainer.vectorize(arr)
            prediction = self.ml_trainer.predict(vector)[0]
            conf = self.ml_trainer.get_probability(vector)[0][prediction]
            self.returnfile.write(str(response.request.url))
            # print("WROTE URL TO FILE")
            self.returnfile.write(",")
            breach = False
            if prediction != 1:
                self.returnfile.write("N,")
            else:
                self.returnfile.write("Y,")
                breach = True
            self.returnfile.write(str(conf * 100))
            self.returnfile.write("\n")
            # print("DATA WRITTEN TO FILE")
            base_url = response.request.url.split('//')[-1].split('/')[0]

            db.threaded_db_add(BaseURL=base_url, FullURL=response.request.url, ScrapyRedirectURL=response.url, FullText=bodyText,
                                  LemmatizedText=lemmatized_text, Breach=breach, conf=conf)

        except Exception as e:
            print("EXCEPTION WHEN DOING MACHINE LEARNING: " + str(e))

        # Crawls next page if flag is on
        if not self.seedOnly:
            # self.BLOCKED_DOMAINS = []
            nextpage = response.xpath("//body//@href").getall()  # uses xpath to extract hrefs from webpages
            for href in nextpage:
                isBlocked = False
                href = response.urljoin(href)
                for link in self.BLOCKED_DOMAINS:
                    if href.lower().find(link) != -1:
                        isBlocked = True
                        break
                    if not isBlocked:
                        # print("FOUND LINK TO CRAWL")
                        request = scrapy.Request(href, self.parse)
                        yield request
