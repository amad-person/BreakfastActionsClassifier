import csv

def read_dict(path):
    """
    Reads Python dictionary stored in a csv file
    """
    dictionary = {}
    for key, val in csv.reader(open(path)):
        dictionary[key] = eval(val)
    return dictionary