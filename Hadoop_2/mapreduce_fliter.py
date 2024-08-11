from mrjob.job import MRJob

class MapReduceFliter(MRJob):
    def mapper (self, _, line):
        data = data[2].strip()