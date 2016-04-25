from scrapy.exceptions import DropItem


class FilterPipeline(object):
    """A pipeline for filtering out pages from certain sites """

    # put all words in lowercase
    sites_to_filter = ['www.amazon.com']

    def process_item(self, item, spider):
        # haven't quite [re]written this one yet.
		for word in self.words_to_filter:
            if word in unicode(item['description']).lower():
                raise DropItem("Contains forbidden word: %s" % word)
        else:
            return item
