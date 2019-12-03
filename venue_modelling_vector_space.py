#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from collections import defaultdict
import jieba


# In[ ]:


from collections import defaultdict
from gensim import corpora, models, similarities


# In[ ]:


stubhub_data = pd.read_csv('Ap_StubHub_Data.csv', sep='\t', lineterminator='\r')


# In[ ]:


stubhub_data.shape


# In[ ]:


stubhub_data = stubhub_data.dropna(subset=['VenueName']) 


# In[ ]:


stubhub_data = stubhub_data.drop(['SourceProductionId', 'ProductionName', 'SourceId', 'SourceName', 'ProductionUrl', 'ProductionDate'], axis = 1)


# In[ ]:


stubhub_data


# In[ ]:


stubhub_data = stubhub_data.rename(columns = {'SourceVenueId': 'stubhub_venue_id', 'VenueName': 'stubhub_venue', 'VenueCity': 'stubhub_city', 'VenueState': 'stubhub_venue_state'})


# In[ ]:


stubhub_data['stubhub_city'] = stubhub_data['stubhub_city'].fillna('#')


# In[ ]:


stubhub_data


# In[ ]:


stubhub_data['stubhub_venue_state'] = stubhub_data['stubhub_venue_state'].fillna('#')


# In[ ]:


stubhub_data['stubhub_venue_desc'] = stubhub_data['stubhub_venue'] + ' ' + stubhub_data['stubhub_city'] + ' ' + stubhub_data['stubhub_venue_state']


# In[ ]:


stubhub_data = stubhub_data[['stubhub_venue_id', 'stubhub_venue_desc']]


# In[ ]:


stubhub_venue_dict = dict(zip(stubhub_data.stubhub_venue_id, stubhub_data.stubhub_venue_desc))


# ## Ticket_Evolution Normalization

# In[ ]:


ticketevolution_data = pd.read_csv('Ap_TicketEvolution_data.csv', sep='\t', lineterminator='\r')


# In[ ]:


ticketevolution_data.shape


# In[ ]:


ticketevolution_data = ticketevolution_data.dropna(subset=['VenueName'])


# In[ ]:


ticketevolution_data = ticketevolution_data.drop(columns = ['Unnamed: 10', 'SourceProductionId', 'ProductionName', 'SourceId', 'SourceName', 'ProductionUrl', 'ProductionDate'], axis = 1)


# In[ ]:


ticketevolution_data.columns


# In[ ]:


ticketevolution_data['ticket_evolution_desc'] = ticketevolution_data['VenueName'] + ' ' + ticketevolution_data['VenueCity'] + ' ' + ticketevolution_data['VenueState']


# In[ ]:


ticketevolution_data


# In[ ]:


ticketevolution_data = ticketevolution_data.rename(columns = {'SourceVenueId': 'ticket_evolution_venue_id'})


# In[ ]:


ticketevolution_data = ticketevolution_data[['ticket_evolution_venue_id', 'ticket_evolution_desc']]


# In[ ]:


ticketevolution_data = ticketevolution_data.astype({'ticket_evolution_desc': 'str'})


# In[ ]:


ticketevolution_venue_dict = dict(zip(ticketevolution_data.ticket_evolution_venue_id, ticketevolution_data.ticket_evolution_desc))


# In[ ]:


ticketevolution_venue_dict


# In[ ]:


def textual_similarity(stubhub, ticketevolution):
    matching_event_dict = defaultdict(list)
    tokens_ticket_events = [jieba.lcut(ticketevolution[y]) for y in ticketevolution]
    counter = 0
    for stubhub_id, stubhub_event in stubhub.items():
        dictionary = corpora.Dictionary(tokens_ticket_events)
        feature_count = len(dictionary.token2id)
        corpus = [dictionary.doc2bow(text) for text in tokens_ticket_events]
        tfidf = models.TfidfModel(corpus)
        new_vec = dictionary.doc2bow(jieba.lcut(stubhub_event))
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_count)
        similiarity = index[tfidf[new_vec]]
        if(max(similiarity) > 0.70):
            arg_max = np.argmax(similiarity)
            ticket_evolution_match = list(ticketevolution.items())[arg_max]
            matching_event_dict['stubhub_id'].append(int(stubhub_id))
            matching_event_dict['stubhub_event'].append(stubhub_event)
            matching_event_dict['ticket_evolution_id'].append(int(ticket_evolution_match[0]))
            matching_event_dict['ticket_evolution_event'].append(ticket_evolution_match[1])
            counter += 1
            print(counter)
    return matching_event_dict
        


# In[ ]:


matching_events = textual_similarity(stubhub_venue_dict, ticketevolution_venue_dict)


# In[ ]:


match_venue_df = pd.DataFrame.from_dict(matching_events)


# In[ ]:


match_venue_df.to_csv('stubhub_ticketevolution_venue_desc_match_all.csv', index= False, header = True)


# In[ ]:


match_venue_df.head()

