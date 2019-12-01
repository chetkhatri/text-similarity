#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import defaultdict
import jieba


# In[2]:


stubhub_data = pd.read_csv("stubhub.csv", sep="\t", lineterminator="\r")


# In[3]:


stubhub_data.shape


# In[4]:


stubhub_data = stubhub_data.dropna(subset=["VenueName"])


# In[5]:


stubhub_event = stubhub_data[["SourceProductionId", "ProductionName"]]
stubhub_event_dict = dict(
    zip(stubhub_event.SourceProductionId, stubhub_event.ProductionName)
)


# In[6]:


stubhub_data = stubhub_data.drop(
    [
        "SourceProductionId",
        "ProductionName",
        "SourceId",
        "SourceName",
        "ProductionUrl",
        "ProductionDate",
    ],
    axis=1,
)


# In[7]:


stubhub_data


# In[8]:


stubhub_data = stubhub_data.rename(
    columns={
        "SourceVenueId": "stubhub_venue_id",
        "VenueName": "stubhub_venue",
        "VenueCity": "stubhub_city",
    }
)


# In[9]:


ticketevolution_data = pd.read_csv("ticketevolution.csv", sep="\t", lineterminator="\r")


# In[10]:


ticketevolution_data.shape


# In[11]:


ticketevolution_data = ticketevolution_data.dropna(subset=["VenueName"])


# In[12]:


ticketevolution_event_data = ticketevolution_data[
    ["SourceProductionId", "ProductionName"]
]
ticketevolution_event_dict = dict(
    zip(
        ticketevolution_event_data.SourceProductionId,
        ticketevolution_event_data.ProductionName,
    )
)


# In[13]:


ticketevolution_data = ticketevolution_data.drop(
    [
        "Unnamed: 10",
        "SourceProductionId",
        "ProductionName",
        "SourceId",
        "SourceName",
        "ProductionUrl",
        "ProductionDate",
    ],
    axis=1,
)


# In[14]:


from collections import defaultdict
from gensim import corpora, models, similarities


# In[15]:


dict_stubhub_venue = stubhub_data["stubhub_venue"].to_list()


# In[16]:


stoplist = set("for a of the and to in".split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in dict_stubhub_venue
]


# In[17]:


dictionary = corpora.Dictionary(texts)


# In[18]:


corpus = [dictionary.doc2bow(text) for text in texts]


# In[19]:


tfidf = models.TfidfModel(corpus)


# In[20]:


keyword = ticketevolution_data["VenueName"][5]


# In[21]:


feature_cnt = len(dictionary.token2id)


# In[22]:


kw_vector = dictionary.doc2bow(keyword.lower().split())


# In[23]:


index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)


# In[24]:


sim = index[tfidf[kw_vector]]


# In[25]:


for i in range(len(sim)):
    if sim[i] > 0.60:
        print("keyword is similar to text %d: %.2f" % (i + 1, sim[i]))


# In[26]:


stubhub_venue = {}
for i in range(len(stubhub_data["stubhub_venue_id"])):
    venue_name = stubhub_data["stubhub_venue"][i]
    venue_id = stubhub_data["stubhub_venue_id"][i]
    stubhub_venue[venue_id] = venue_name


# In[27]:


ticketevolution_venue = {}
for i in range(len(ticketevolution_data["SourceVenueId"])):
    venue_name = ticketevolution_data["VenueName"][i]
    venue_id = str(ticketevolution_data["SourceVenueId"][i])
    ticketevolution_venue[venue_id] = venue_name



def gensimCalculation(stubhub, ticketevolution):
    matching_event_dict = defaultdict(list)
    tokens_ticket_events = [jieba.lcut(ticketevolution[y]) for y in ticketevolution]
    counter = 0
    for stubhub_id, stubhub_event in stubhub.items():
        dictionary = corpora.Dictionary(tokens_ticket_events)
        feature_count = len(dictionary.token2id)
        corpus = [dictionary.doc2bow(text) for text in tokens_ticket_events]
        tfidf = models.TfidfModel(corpus)
        new_vec = dictionary.doc2bow(jieba.lcut(stubhub_event))
        index = similarities.SparseMatrixSimilarity(
            tfidf[corpus], num_features=feature_count
        )
        similiarity = index[tfidf[new_vec]]
        if max(similiarity) > 0.70:
            arg_max = np.argmax(similiarity)
            ticket_evolution_match = list(ticketevolution.items())[arg_max]
            matching_event_dict["stubhub_id"].append(int(stubhub_id))
            matching_event_dict["stubhub_event"].append(stubhub_event)
            matching_event_dict["ticket_evolution_id"].append(
                int(ticket_evolution_match[0])
            )
            matching_event_dict["ticket_evolution_event"].append(
                ticket_evolution_match[1]
            )
            counter += 1
            print(counter)
    return matching_event_dict


# In[30]:


matching_events = gensimCalculation(stubhub_event_dict, ticketevolution_event_dict)


# In[31]:


vector_match_df = pd.DataFrame.from_dict(matching_events)


# In[32]:


vector_match_df.to_csv("stubhub_ticketnetwork_match.csv", index=True, header=True)


