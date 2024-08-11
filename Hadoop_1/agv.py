from mrjob.job import MRJob

class MapReduceAvg(MRJob):
    def mapper(self, _, line):
        data = line.split(',')
        status_type = data[1].strip()
        num_reactions = data[3].strip()
        try :

            yield status_type, float(num_reactions)
        except: pass

    def reducer(self, key, values):  
        lval = list(values) 
        yield key, round(sum(lval)/len(lval),2)


if (__name__ == "__main__" ):
    MapReduceAvg.run()