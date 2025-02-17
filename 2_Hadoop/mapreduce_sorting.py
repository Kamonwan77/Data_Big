from mrjob.job import MRJob

class  MapReduceSorted(MRJob):
    def mapper(self, _, line):

        if 'status_id' not in line:

            data = line.split(',')
            
            date = data[2].strip()

            year = date.split(" ")[0].split('/')[2]

            num_reactions = data[3].strip()
            
            if int(num_reactions) > 3000 :
                yield year , num_reactions

    def reducer(self , key , values):
        sorted_value = sorted(values)
        yield key , sorted_value

if __name__ == '__main__':
    MapReduceSorted.run()