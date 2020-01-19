# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from breachSpider.machineLearning.MachineLearning import MachineLearning


class BreachspiderPipeline(object):
    def close_spider(self,spider):
        # print(spider.lemmatized_documents)
        # print(spider.labels)
        spider.file.close()
        # ml_trainer = MachineLearningTrainer(spider.file_name)
        # ml_trainer.run()
