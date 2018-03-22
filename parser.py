import pandas as pd

class Review(object):
    """docstring for Review"""
    def __init__(self):
        super(Review, self).__init__()

    @staticmethod
    def parse(xml_data):
        current_key = None
        data = {}
        for line in xml_data:
            if line.startswith("</") and line.endswith(">"):
                current_key = None
            elif line.startswith("<") and line.endswith(">"):
                current_key = line[1:-1]
                if current_key not in data:
                    data[current_key] = []
            else:
                data[current_key].append(line)
        return data


class AmazonReviewsParser(object):
    """docstring for AmazonReviewsParser"""
    def __init__(self, filepath):
        super(AmazonReviewsParser, self).__init__()
        self.filepath = filepath

    @staticmethod
    def parse(filepath):
        review = []
        reviews = []
        with open(filepath, 'r', encoding="ISO-8859-1") as f:
            for line in f:
                line = line.rstrip()
                if line == "<review>":
                    pass
                elif line == "</review>":
                    reviews.append(Review.parse(review))
                    review = []
                else:
                    review.append(line)
        return pd.DataFrame(reviews)



