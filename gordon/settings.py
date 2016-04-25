# Scrapy settings for gordon project

SPIDER_MODULES = ['gordon.spiders']
NEWSPIDER_MODULE = 'gordon.spiders'
DEFAULT_ITEM_CLASS = 'gordon.items.Website'

#ITEM_PIPELINES = {'gordon.pipelines.FilterPipeline': 1}
