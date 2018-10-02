#!/usr/bin/env python3
#Author: Zhu Yong-kai (yongkai_zhu@hotmail.com)
"""
This is a program used to get the doi value from a article in PDF format,
and then it will execute the basic search using the doi as the only key.
The results returned from the website will be saved as yaml format which is
a nice and readable format.
"""
import os
import re
import subprocess
import argparse
import requests
import sys

import yaml
from bs4 import BeautifulSoup


class ParseHtml:
    """
    Parse the web of adsabs and get the information (like authors and abstract) of
    a paper.
    """
    def __init__(self, url, doi):
        self.url = url
        self.doi = doi
        self.link = url + self.doi

    def urlRequest(self):
        r = requests.get(self.link)
        if r.status_code == 200:
            print("The request was successful.")
        else:
            raise ValueError('Response error: the returned error code is %s'% r.status_code)
        self.r = r.text
        return self.r

    def parseSoup(self):
        r = self.urlRequest()
        soup = BeautifulSoup(r, "lxml")
        articleInfo = {}
        for tag in soup.html.body.find_all('b'):
            key = str(tag.string)
            value = []
            if key == 'Bibtex entry for this abstract':
                bibtex_url = str(tag.a['href'])
                articleInfo['bibtex_url'] = bibtex_url
            elif key ==  'Preferred format for this abstract':
                preferred_format_url = str(tag.a['href'])
                articleInfo['preferred_format_url'] = preferred_format_url
            else:
                i = tag.parent.parent.find_all('td')
                for j in i[2]:
                    if len(j.string) > 2:
                        k = str(j.string.replace('\xa0', ' '))
                        value.append(k)
                articleInfo[key] = value
        abstract = re.search('Abstract</h3>(.*)[\s\S]*?(<hr>|$)', r)
        abstract = abstract.group().replace('Abstract</h3>','').\
            replace('<hr>','').strip()
        articleInfo['Abstract:'] = str(abstract)
        return articleInfo


def getDoi(fileName):
    """
    Using the subprocess module to execute a command-line process with pdftotext
    to convert pdf file to text file. From the text file, it will be easy to get
    the doi of the paper.
    """
    retcode, output = subprocess.getstatusoutput("pdftotext" + " " + \
                                                fileName)
    if retcode == 0:
        textName = os.path.splitext(fileName)[0]+ ".txt"
        f = open(textName)
        for eachline in f:
            ret = (re.search('10\.[0-9]*/(.+?)(\s|$)', eachline))
            if ret is not None:
                doi = ret.group().rstrip()
                break
        if ret is None:
            raise ValueError("Ops! The doi does not exist")
    else:
        print(output)
        raise RuntimeError("pdftotext failed")
    return doi


def main():
    parser = argparse.ArgumentParser(
        description="Obtain the doi value from a published article\
        with a format of pdf")
    parser.add_argument("-i", "--inputfile", dest="inputfile",
                        required=True,
                        help="Input the pdf file name")
    parser.add_argument("-o", "--outpath", dest="outpath",
                        help="enter the path where you want to save your result")
    args = parser.parse_args()
    doicode = getDoi(args.inputfile)
    print("The paper's doi: "+doicode)
    url = "http://adsabs.harvard.edu/doi/"
    ph = ParseHtml(url=url, doi=doicode)
    articleInfo = ph.parseSoup()
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    if args.outpath is None:
        f = open(os.path.splitext(args.inputfile)[0] + '.yaml','w')
    else:
        f = open(args.outpath +
                 os.path.splitext(os.path.split(args.inputfile)[1])[0] + '.yaml','w')
    yaml.dump(articleInfo, f)


if __name__ == "__main__":
    main()
