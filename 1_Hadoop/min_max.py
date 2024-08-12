from mrjob.job import MRJob

class MapReduceMax(MRJob):

    def mapper(self, _, line):
        status_type = data[1].strip()

        if status_type == 'photo':
            yield (year, 'Photo') , 1
                   
        elif status_type == 'video':
            yield (year, 'Video'), 1
                   
        elif status_type == 'link':
            yield (year, 'Link'), 1
                   
        elif status_type == 'status':
            yield (year, 'Status'), 1

    def reducer_count(self, key, value):
        yield key[0],(sum(value), key[1])
    

    def reducer_max(self, key, value):
        yield key, max(value)  
    
    def steps(self):
        return [MRStep(mapper = self.mapper, \
                       
                reducer = self.reducer_count), \
                MRStep(reducer = self.reducer_max)]


if (__name__ == "__main__"):
    MapReduceMax.run()