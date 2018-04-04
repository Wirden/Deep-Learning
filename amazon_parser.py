import pandas as pd
import datetime


class Review(object):
    """docstring for Review"""
    def __init__(self):
        super().__init__()

    KEY_MULTI_ENTRY = ['unique_id', 'product_type']

    MONTHS_MAP = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    @staticmethod
    def parse_date(date_str: str):
        try:
            month_str, day_str, year_str = date_str.split()
            return datetime.datetime(int(year_str), Review.MONTHS_MAP[month_str], int(day_str[:-1]))
        except ValueError:
            return None

    @staticmethod
    def parse(xml_data: list):
        current_key = None
        data = {}
        for line in xml_data:
            if line.startswith("</") and line.endswith(">"):
                current_key = None
            elif line.startswith("<") and line.endswith(">"):
                current_key = line[1:-1]
                if current_key not in data:
                    if current_key in Review.KEY_MULTI_ENTRY:
                        data[current_key] = []
                    else:
                        data[current_key] = None
            else:
                if current_key in Review.KEY_MULTI_ENTRY:
                    data[current_key].append(line)
                elif current_key == 'date':
                    data[current_key] = Review.parse_date(line)
                else:
                    data[current_key] = line
        pos_helpful, total_helpful = Review.helpful2score(data['helpful'])
        data['pos_helpful'] = pos_helpful
        data['total_helpful'] = total_helpful
        return data

    @staticmethod
    def helpful2score(helpful_list: list):
        try:
            positive, total = helpful_list.split(' of ')
            return int(positive), int(total)
        except ValueError:  # d/D can't be parsed as ints
            return 0.0, 0.0


class AmazonReviewsParser(object):
    """docstring for AmazonReviewsParser"""
    def __init__(self, filepath):
        super().__init__()
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
