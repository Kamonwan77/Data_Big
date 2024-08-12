from mrjob.job import MRJob
from mrjob.step import MRStep

class MapReduceTopN(MRJob):
    def mapper(self, _, line):
        if 'status_id' not in line:
            data = line.split(',')  # Data is a list of values in each line of a file
            date = data[2].strip()
            year = date.split(' ')[0].split('/')[2]
            status_type = data[1].strip()  # Get the status type
            yield (year, status_type), 1

    def reducer_count(self, key, values):
        yield key[0], (sum(values), key[1])  # Count the number of each status type per year

    def reducer_topN(self, key, values):
        N = 2
        # Sort the values by count in descending order and then by status type
        topN = sorted(values, reverse=True, key=lambda x: (x[0], x[1]))[:N]
        yield key, topN

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer_count),
            MRStep(reducer=self.reducer_topN)
        ]

if __name__ == '__main__':
    MapReduceTopN.run()