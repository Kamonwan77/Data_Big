from mrjob.job import MRJob

class  MapReduceBinning(MRJob):
    def mapper(self, _, line):

        if 'status_id' not in line:

            data = line.split(',')
            
            datetime = data[2].strip()

            year = datetime.split(' ')[0].split('/')[2]

            status_type = data[1].strip()

            if year == '2018' :
                yield status_type , data 

if __name__ == '__main__':
    MapReduceBinning.run()