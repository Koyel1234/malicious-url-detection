#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import pickle

import re
from urllib import parse
from urllib.parse import urlparse
import tldextract #to extract subdomain,domain,tld


# In[3]:


## load pickle file of baseline model
with open('Baseline(Logistic).pickle','rb') as new_df:
    model=pickle.load(new_df)


# In[4]:


from pywebio.input import *
from pywebio.output import *


# In[5]:


app=Flask(__name__)

def result():
    
    link_to_check=input('Enter URL you want to check',type=TEXT)
    
    def IpAdress(link_to_check):
        if re.match(r'^(http|https)://\d+\.\d+\.\d+\.\d+\.*',link_to_check) is None:
            return 0
        else:
            return 1
    
    def presence_of(link_to_check,character):
        if link_to_check.count(character)==0: #means character not present
            return 0
        else:
            return 1
        
    def NumDashInHostname_in_link(link_to_check):
        return (urlparse(link_to_check).netloc).count('-')    
        

    UrlLength=len(link_to_check)
    NumDash=link_to_check.count('-')
    NumDots=link_to_check.count('.')
    NumUnderscore=link_to_check.count('_'), 
    NumPercent=link_to_check.count('%')
    NumAmpersand=link_to_check.count('&')
    NumDashInHostname=NumDashInHostname_in_link(link_to_check)
    NumNumericChars=sum(c.isdigit() for c in link_to_check)
    IpAdress=IpAdress(link_to_check)
    NumQueryComponents=len(dict(parse.parse_qs(parse.urlsplit(link_to_check, allow_fragments=False).query)))
    HostnameLength=len(urlparse(link_to_check).netloc)      
    PathLength=len(urlparse(link_to_check).path)
    QueryLength=len(urlparse(link_to_check).query)
    SubdomainLevel=len(tldextract.extract(link_to_check).subdomain.split('.'))
    NumHash=link_to_check.count('#')
    AtSymbol=presence_of(link_to_check,'@')
    TildeSymbol=presence_of(link_to_check,'~')
    NoHttps=presence_of((urlparse(link_to_check).scheme),'https')
    SubdomainLevel=len(tldextract.extract(link_to_check).subdomain.split('.'))
    DoubleSlashInPath=presence_of((urlparse(link_to_check).path),"//")



    
    dict1={"UrlLength":UrlLength,'NumDash':NumDash, "NumDots":NumDots, 'NumUnderscore':NumUnderscore[0], 'NumPercent':NumPercent, 'NumAmpersand':NumAmpersand, 'NumDashInHostname':NumDashInHostname, 'NumNumericChars':NumNumericChars, 'IpAdress':IpAdress, 'NumQueryComponents':NumQueryComponents, 'HostnameLength':HostnameLength, 'PathLength':PathLength, 'QueryLength':QueryLength,'NumHash':NumHash,'AtSymbol':AtSymbol,'TildeSymbol':TildeSymbol,'NoHttps':NoHttps,'SubdomainLevel':SubdomainLevel,'DoubleSlashInPath':DoubleSlashInPath}   
    df1=pd.DataFrame(dict1, index=[1])
    
    if model.predict(df1)[0] == 0:
        put_text("The link seems not malicious.")
    else:
        put_text("link seems malicious.")

if __name__=='__main__':
    result()
    
    
app.add_url_rule('/tool','webio_view',webio_view(result),methods=['GET','POST','OPTIONS'])    

