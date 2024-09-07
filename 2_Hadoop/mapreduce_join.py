from mrjob.job import MRJob

class  MapReduceInner(MRJob):
    def mapper(self, _, line):
        if 'status_id' not in line:

            data = line.split(',')
            
            fbID = data[1]

            yield fbID, data

    def reducer(self , key , values):
        fb2 = []
        fb3 = []
        for v in values:
            if v[0] == 'FB2':
                fb2.append(v)
            elif v[0] == 'FB3':
                fb3.append(v)
        #Inner join
        for i in fb2:
            for j in fb3:
                yield None , (i+j)

if __name__ == '__main__':
    MapReduceInner.run()