from scrapy.item import Item, Field

class Webpage(Item):
	url = Field()
	score = Field()
	
class WordCount(Item):
	word = Field()
	count = Field()

#class Wobsite(Item):
#    name = Field()
#    description = Field()
#    url = Field()
