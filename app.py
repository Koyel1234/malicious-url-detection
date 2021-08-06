import pandas as pd
import numpy as np

from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server

import pickle

import re
from urllib import parse
from urllib.parse import urlparse
import tldextract #to extract subdomain,domain,tld



app=Flask(__name__)



with open('Phishing_Model.pickle','rb') as new_df:
    loaded_model=pickle.load(new_df)


with open('Isolation_Forest_Score.pickle','rb') as file:
    clf=pickle.load(file)


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
    df1.to_csv("phishing_test_1.csv", header=None, index=False)
    x1=df1.values
    predictions=clf.predict(x1)
    clf.decision_function(x1)
    
    
    if NumDots>=4:
        NumDots_GrtrEql_4_EDA=1
    else:
        NumDots_GrtrEql_4_EDA=0

    if NumDash<=1:
        NumDash_LsEql_1_EDA=1
    else:
        NumDash_LsEql_1_EDA=0



    if NumQueryComponents==0:
        NumQueryComponents_EqlTo_0_EDA=1
    else:
        NumQueryComponents_EqlTo_0_EDA=0



    if NumNumericChars==0:
        NumNumericChars_EqlTo_0_EDA=1
    else:
        NumNumericChars_EqlTo_0_EDA=0


    if UrlLength<35:
        UrlLength_WOE=-0.462714
    elif 35<=UrlLength<60:
        UrlLength_WOE=0.492540
    elif 60<=UrlLength<70:
        UrlLength_WOE=0.085523
    elif 70<=UrlLength<130:
        UrlLength_WOE=-0.571443
    else:
        UrlLength_WOE=0.211513 


    if NumDash<2:
        NumDash_WOE=0.506946
    elif 2<=NumDash<3:
        NumDash_WOE=-0.216327
    elif 3<=NumDash<6:
        NumDash_WOE=-1.194891
    else:
        NumDash_WOE=-3.685051


    if NumDots<2:
        NumDots_WOE=-1.319003
    elif 2<=NumDots<3:
        NumDots_WOE=-0.019816
    elif 3<=NumDots<4:
        NumDots_WOE=0.292073
    else:
        NumDots_WOE=1.422297


    if NumUnderscore[0]<1:
        NumUnderscore_WOE=0.081354
    elif 1<=NumUnderscore[0]<2:
        NumUnderscore_WOE=-0.192783
    else:
        NumUnderscore_WOE=-0.869469  


    if NumAmpersand<1:
        NumAmpersand_WOE=0.114997
    else:
        NumAmpersand_WOE=-1.146679    



    if NumDashInHostname<1:
        NumDashInHostname_WOE=-0.096186
    else:
        NumDashInHostname_WOE=1.000892  


    if NumNumericChars<2:
        NumNumericChars_WOE=0.707812
    elif 2<=NumNumericChars<5:
        NumNumericChars_WOE=-0.494543
    elif 5<=NumNumericChars<8:
        NumNumericChars_WOE=-1.830517
    elif 8<=NumNumericChars<9:
        NumNumericChars_WOE=-3.127076
    elif 9<=NumNumericChars<18:
        NumNumericChars_WOE=-0.777151
    else:
        NumNumericChars_WOE=1.319388    




    if SubdomainLevel<1:
        SubdomainLevel_WOE=0.062771
    else:
        SubdomainLevel_WOE=-0.057979



    if HostnameLength<11:
        HostnameLength_WOE=-0.705058
    elif 11<=HostnameLength<20:
        HostnameLength_WOE=-0.120158
    elif 20<=HostnameLength<26:
        HostnameLength_WOE=0.080934
    elif 26<=HostnameLength<30:    
        HostnameLength_WOE=0.484893
    else:
        HostnameLength_WOE=1.202948




    if PathLength<8:
        PathLength_WOE=-0.873511
    elif 8<=PathLength<16:
        PathLength_WOE=-0.203832
    elif 16<=PathLength<46:
        PathLength_WOE=0.402728
    elif 46<=PathLength<60:
        PathLength_WOE=-0.117753
    else:
        PathLength_WOE=-0.833557



    if QueryLength<6:
        QueryLength_WOE=0.222194
    elif 6<=QueryLength<30:
        QueryLength_WOE=-1.813158
    else:
        QueryLength_WOE=-0.491447   


    ISF_Score=clf.decision_function(x1)    


    dict2={"UrlLength":UrlLength,'NumDash':NumDash, "NumDots":NumDots, 'NumUnderscore':NumUnderscore[0], 'NumPercent':NumPercent,       'NumAmpersand':NumAmpersand, 'NumDashInHostname':NumDashInHostname, 'NumNumericChars':NumNumericChars, 'IpAddress':IpAdress,   'NumQueryComponents':NumQueryComponents, 'HostnameLength':HostnameLength, 'PathLength':PathLength, 'QueryLength':QueryLength, 'NumDots_GrtrEql_4_EDA':NumDots_GrtrEql_4_EDA, 'NumDash_LsEql_1_EDA':NumDash_LsEql_1_EDA, 'NumQueryComponents==0_EDA':NumQueryComponents_EqlTo_0_EDA, 'NumNumericChars==0_EDA':NumNumericChars_EqlTo_0_EDA,'UrlLength_WOE':UrlLength_WOE, 'NumDash_WOE':NumDash_WOE, 'NumDots_WOE':NumDots_WOE,'NumUnderscore_WOE':NumUnderscore_WOE, 'NumAmpersand_WOE':NumAmpersand_WOE,'NumDashInHostname_WOE':NumDashInHostname_WOE, 'NumNumericChars_WOE':NumNumericChars_WOE,'SubdomainLevel_WOE':SubdomainLevel_WOE, 'HostnameLength_WOE':HostnameLength_WOE,'PathLength_WOE':PathLength_WOE,'QueryLength_WOE':QueryLength_WOE, 'ISF_Score':ISF_Score}
    df2=pd.DataFrame(dict2, index=[1])
    

    if loaded_model.predict(df2)[0] == 0:
        put_text("The link seems not malicious.")
    else:
        put_text("link seems malicious.")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    
    start_server(result, port=args.port)
    
