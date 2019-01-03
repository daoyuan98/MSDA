class Dataset(object):

    def __init__(self, train, test, train_label, test_label, name):
        self.train = DataBatch(train, train_label)
        self.test  = DataBatch(test, test_label)
        self.name = name

    def __len__(self):
        return len(self.train)

    def __str__(self):
        return self.name


class DataBatch():

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
