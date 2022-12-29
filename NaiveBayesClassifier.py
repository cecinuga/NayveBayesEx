from collections import Counter
import copy
import csv
import numpy as np

dataset_path = '../datasets/digits.csv'
split_ratio = 1

class NaiveBayes(object):
    def __init__(self, filename: str, test_size=80):
        self.test_size = test_size
        self.filename = filename
        self.freq = np.ndarray
        self.y_count = np.ndarray
        self.dataset = np.ndarray
        self.train_set = np.ndarray
        self.test_set = np.ndarray
        self.prior = np.ndarray
        self.accuracy = 0
    
    def load_dataset(self):
        lines = csv.reader(open(self.filename, 'r'))
        self.dataset = np.array(list(lines), dtype=np.longdouble)

    def split_dataset(self):
        np.random.shuffle(self.dataset)
        size = int(len(self.dataset)*self.test_size/100)
        self.train_set, self.test_set = self.dataset[:size, :], self.dataset[size:, :]

    def prior_prob(self):
        self.y_count = np.unique(self.train_set[:, -1], return_counts=True)
        self.prior = self.y_count[1]/len(self.train_set)

    def freq_table(self): 
        counter = [[np.array([]) for j in range(len(self.train_set[0])-1)] for i in self.prior]  
        support = copy.deepcopy(counter)
        #counter = [np.array(np.unique(self.train_set[:, col], return_counts=True), dtype=np.longdouble) for col in range(self.train_set.shape[1])]
        """crea la frequency table"""
        for row in range(len(self.train_set)):
            idx = np.where(self.y_count[0] == self.train_set[row, -1])
            for col in range(len(self.train_set[0])-1):
                counter[idx[0][0]][col] = np.append(counter[idx[0][0]][col], self.train_set[row][col])
                support[idx[0][0]][col] = np.unique(counter[idx[0][0]][col], return_counts=True)

        for prior in range(len(support)):
            for col in range(len(support[prior])):
                support[prior][col] = list(support[prior][col])
                support[prior][col][1] = support[prior][col][1] / self.y_count[1][prior]
        self.freq = support

    def bayes(self, row: np.ndarray):
        prob = copy.deepcopy(self.prior)
        for col in range(len(row)):
            for j in range(len(self.freq)):
                idx = np.where(self.freq[j][col][0] == row[col])
                #print(self.freq[j][col][1][idx[0]])
                if(len(idx[0])>0):
                    prob[j] = prob[j] * self.freq[j][col][1][idx[0]]
        return np.where(prob == np.amax(prob))[0][0]

    def predict(self):
        pos = 0
        self.prior_prob()
        self.freq_table()
        for row in self.test_set:
            predicted = self.bayes(row[0:-1])
            #print("Original: {0}, Predicted: {1}".format(row[-1], predicted))
            if row[-1] == predicted:
                pos = pos + 1
        print('Number of Positive Predictions on {0} row, is {1}'.format(len(self.test_set), pos))
        self.accuracy = (100*pos)/len(self.test_set)

def main():
    model = NaiveBayes(dataset_path, 80)
    model.load_dataset()
    model.split_dataset()
    print("Train_Set len: {0}, Test_Set len: {1}".format(len(model.train_set), len(model.test_set)))
    model.predict()
    print('Accuracy: ', model.accuracy, '%')
if __name__ == '__main__':
    main()