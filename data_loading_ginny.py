import pickle, os
from preprocessing.preprocessing import PreProcessing
from GEODataCollection.reading_geo_data import GetGeoData
from PubmedDataCollection.reading_article_data import GetArticleData
from Rec_configuraiton import Rec_configuration

class DataLoading:

    def __init__(self):
        self.rec_conf = Rec_configuration()
        #self.pp = PreProcessing()
        #Geo data processing
        #self.geo_data = None
        self.geo_title, self.geo_summary = None, None
        self.citation_data = None
        #Article data processing 
        self.article_title, self.article_abstract = None, None
        self.initial_checking_loading_processing()

    def get_all_details(self):
        return self.article_title, self.article_abstract, \
               self.geo_title, self.geo_summary, self.citation_data
               
    def initial_checking_loading_processing(self, word_pr= False ):
        print('Loading all geo related preprocessed datasets')
        self.geo_title = pickle.load(open(self.rec_conf.file_address[2], 'rb'))
        self.geo_summary = pickle.load(open(self.rec_conf.file_address[3], 'rb'))
        self.citation_data = pickle.load(open(self.rec_conf.file_address[4], 'rb'))
        print('Loaded all geo related processed datasets')
        
        if word_pr:
            print('Preprocessing geo titles')
            self.geo_title = self.pp.preprocess_all(self.geo_title)
            pickle.dump(self.geo_title, open(self.rec_conf.file_address[2], 'wb'), protocol=4)
            print('Processed and save geo titles')
    
            print('Preprocessing geo summaries')
            self.geo_summary = self.pp.preprocess_all(self.geo_summary)
            pickle.dump(self.geo_summary, open(self.rec_conf.file_address[3], 'wb'), protocol=4)
            print('Processed and save geo summeries')

        
        print('Loading all title and abstract')
        self.article_title = pickle.load(open(self.rec_conf.file_address[0], 'rb'))
        self.article_abstract = pickle.load(open(self.rec_conf.file_address[1], 'rb'))
        print('Loaded all title and abstract')
        
        
        if word_pr:
            print('Preprocessing article titles')
            self.article_title = self.pp.preprocess_all(self.article_title)
            pickle.dump(self.article_title, open(self.rec_conf.file_address[0], 'wb'), protocol=4)
            print('Processed and save article titles')
    
            print('Preprocessing article abstracts')
            self.article_abstract = self.pp.preprocess_all(self.article_abstract)
            pickle.dump(self.article_abstract, open(self.rec_conf.file_address[1], 'wb'), protocol=4)
            print('Processed and save article abstracts')
            
        
        
        
    """
    def initial_checking_loading_processing(self):
        if not os.path.exists(self.rec_conf.file_address[2]) or not os.path.exists(self.rec_conf.file_address[2]) or \
            not os.path.exists(self.rec_conf.file_address[4]):
            print('Loading raw datasets')
            geo_data = GetGeoData(self.rec_conf.geo_file_address)
            self.citation_data = geo_data.collect_geo_citations()
            pickle.dump(self.citation_data, open(self.rec_conf.file_address[4], 'wb'), protocol=4)
            print('Saved citation details')

            self.geo_title, self.geo_summary = geo_data.collect_geo_title_n_summary()
            print('geo titles = {}, summary={}'.format(len(self.geo_title), len(self.geo_summary)))

            geo_data = None

            print('Preprocessing geo titles')
            self.geo_title = self.pp.preprocess_all(self.geo_title)
            pickle.dump(self.geo_title, open(self.rec_conf.file_address[2], 'wb'), protocol=4)
            print('Processed and save geo titles')

            print('Preprocessing geo summaries')
            self.geo_summary = self.pp.preprocess_all(self.geo_summary)
            pickle.dump(self.geo_summary, open(self.rec_conf.file_address[3], 'wb'), protocol=4)
            print('Processed and save geo summeries')

        else:
            print('Loading all geo related preprocessed datasets')
            self.geo_title = pickle.load(open(self.rec_conf.file_address[2], 'rb'))
            self.geo_summary = pickle.load(open(self.rec_conf.file_address[3], 'rb'))
            self.citation_data = pickle.load(open(self.rec_conf.file_address[4], 'rb'))
            print('Loaded all geo related processed datasets')
        
        if not os.path.exists(self.rec_conf.file_address[0]) or not os.path.exists(self.rec_conf.file_address[1]):
            #Article data processing
            #self.article_title, self.article_abstract = \
            #                           GetArticleData(self.article_address).collect_article_title_n_summary()
            print('Loading raw title and abstract')
            self.article_title = pickle.load(open(self.rec_conf.article_title_add, 'rb'))
            self.article_abstract = pickle.load(open(self.rec_conf.article_abstract_add, 'rb'))
            print('article titles = {}, abstract={}'.format(len(self.article_title), len(self.article_abstract)))
        
            print('Preprocessing article titles')
            self.article_title = self.pp.preprocess_all(self.article_title)
            pickle.dump(self.article_title, open(self.rec_conf.file_address[0], 'wb'), protocol=4)
            print('Processed and save article titles')

            print('Preprocessing article abstracts')
            self.article_abstract = self.pp.preprocess_all(self.article_abstract)
            pickle.dump(self.article_abstract, open(self.rec_conf.file_address[1], 'wb'), protocol=4)
            print('Processed and save article abstracts')
        else:
            print('Loading all title and abstract')
            self.article_title = pickle.load(open(self.rec_conf.file_address[0], 'rb'))
            self.article_abstract = pickle.load(open(self.rec_conf.file_address[1], 'rb'))
            print('Loaded all title and abstract')
    
    '''def initial_processed_data_loading(self):
        print('Loading processed data')
        self.article_title = self.rnw_pickle.read_pickle_file(self.file_address[0])
        self.article_abstract = self.rnw_pickle.read_pickle_file(self.file_address[1])
        self.geo_title = self.rnw_pickle.read_pickle_file(self.file_address[2])
        self.geo_summary = self.rnw_pickle.read_pickle_file(self.file_address[3])
        self.citation_data = self.rnw_pickle.read_pickle_file(self.file_address[4])'''


    '''def initial_text_processing(self, file_address):
        print('Processing text')
        file_address = file_address
        self.article_title = self.pp.preprocess_all(self.article_title)
        self.rnw_pickle.write_pickle_file(self.article_title, file_address[0])

        self.article_abstract = self.pp.preprocess_all(self.article_abstract)
        self.rnw_pickle.write_pickle_file(self.article_abstract, file_address[1])
        
        self.geo_title = self.pp.preprocess_all(self.geo_title)
        self.rnw_pickle.write_pickle_file(self.geo_title, file_address[2])

        self.geo_summary = self.pp.preprocess_all(self.geo_summary)
        self.rnw_pickle.write_pickle_file(self.geo_summary, file_address[3])
        print('Saved all processed text')'''


    '''def checking_data(self):
        data_processed = True
        for file_name in self.file_address:
            if not self.rnw.is_file_exist(file_name):
                data_processed = False
                break
        if not data_processed:
            self.initial_raw_data_loading_n_processing()
        else:
            self.initial_processed_data_loading()'''
       """

def main():
    x = DataLoading() 


if __name__=='__main__':
    main()
