# Copyright (C) 2019 Kasisto, Inc.

from collections import defaultdict, Counter
from lxml import etree
from enum import Enum
import numpy as np
import pandas as pd
from nltk.util import skipgrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import itertools as it
import argparse

def token_list_to_token(token_list):
	return "_".join(sorted(token_list))
	

class TokenGram:
	def __init__(self, token_list, k):
		self.already_found = set()
		combos = it.combinations(token_list, k)
		for combo in combos:
			cndt = token_list_to_token(combo)
			if cndt not in self.already_found:
				self.already_found.add(cndt)
		self._grams = list(self.already_found)
		
		# If there are no combinations of length "k", then handle solo 1 and 2 Ks
		if len(self._grams) == 0:
			if len(token_list) == 1:
				self._grams = ["x{}x".format(token_list[0])]
			if len(token_list) == 2:
				self._grams = ["x{}_{}x".format(token_list[0], token_list[1])]
		
	def get_token(self):
		token =  " ".join(self._grams)
		return token
		
	def __len__(self):
		return len(self._grams)
	
	def __str__(self):
		return str(self._grams)
	
	def __repr__(self):
		return self.__str__()
		

def tfidf(docs, labels, arg_min_df=1, arg_max_df=0.75, n_gram=(1,1), top_n=25):
	"""
    term frequency, inverse document frequency

    for each document, get top N terms
    """
	
	print("======================================================================")
	print("CALCULATE TFIDF")
	print("======================================================================")
	
	tf_idf = TfidfVectorizer(min_df=arg_min_df, max_df=arg_max_df, ngram_range=n_gram)
	
	# Max df is proportion of documents that frequency appears in, as cutoff for considering a stop word
	response = tf_idf.fit_transform(docs)
	
	feature_array = np.array(tf_idf.get_feature_names())
	with open("feats_check.txt", 'w') as filename:
		for line in tf_idf.get_feature_names():
			filename.write(line)
			filename.write("\n")
			
	results = []
	for i, qa_label in enumerate(labels):
		# assumes enumeration is consistent with docs
		top_qas = top_feats_in_doc(response, feature_array, i, top_n)
		results.append((qa_label, top_qas))
		# print(qa_label)
		# print(top_qas)
		# print("\n")
	return results


def remove_slots(child_text):
	types = ["{@PaymentType", "{@MerchantType", "{@CountryType", "{@PaymentDest"]
	return " ".join(remove_ending_brace(token) for token in child_text.split() if token not in types)


def remove_ending_brace(token):
	if token[-1] == "}":
		return token[0:-1]
	else:
		return token

DELIMS = [("<", "fstart_"), (">", "_fend"), ("'", "_apo_"), ("-", "_hyp_"), (".", "_dot_"), (",", "_ca_"), ("\"", "_quo_"), ("“", "_quo_"), ("\'", "_apo_"), (")", '_rp_'), ("(", "_lp"), ("/", "_fsl_"), ("\\", "_bsl_"), ("=", "_eq_"), ("@", "_aat_"), ("$", "_dol_"), ("%", "_per_"), ("*", "_star_"), ("#", "_hash_"), ("+", "_plus_"), ("^", "_car_"), ("!", "_excl_"), ("?", "_quest_"), ("&", "_aand_")]
# DELIMS = []

RETURN_DELIMS = {y: x for (x,y) in DELIMS}

def make_replacements(original):
	for dx, dy in DELIMS:
		original = original.replace(dx, dy)
	return original


def return_replacements(original):
	for dx, dy in RETURN_DELIMS.items():
		original = original.replace(dx, dy)

	return original


def top_tfidf_feats(row, features, top_n=25):
	''' Get top n tfidf values in row and return them with their corresponding feature names.'''
	topn_ids = np.argsort(row)[::-1][:top_n]
	top_feats = [(return_replacements(features[i]), row[i]) for i in topn_ids]
	df = pd.DataFrame(top_feats)
	df.columns = ['feature', 'tfidf']
	return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
	''' Top tfidf features in specific document (matrix row) '''
	row = np.squeeze(Xtr[row_id].toarray())
	return top_tfidf_feats(row, features, top_n)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="calculate tfidf")
	parser.add_argument("--app",dest="APP", default="liv", help="URL to portal server, including port")
	parser.add_argument("--type",dest="TYPE", default="all", help="which type")
	parser.add_argument("--suffix",dest="SUFFIX", default="5",help="suffix of file") 
	parser.add_argument("--out",dest="OUT", default="test", help="distinguishing feature of name of output text file")
	parser.add_argument("--compare", dest="COMPARE", default=None, help="compare list, enumerated in file")
	parser.add_argument("--top_n", dest="TOP_N", default=15, help="how many per faq")
	parser.add_argument("--n_gram", dest="N_GRAM", default=1, help="how many n-gram")
	parser.add_argument("--k=", dest="K", default=1, help="how many pairwise token grams")
	args = parser.parse_args()

	COMPARE = None
	if args.COMPARE is not None:
		COMPARE = ["VpaDynamicallyCustomizable.PaymentFees", "VpaDynamicallyCustomizable.PaymentExecution"]


	if args.TYPE == "getAnswer":
		file_name = "{}/temp.txt.VpaGetAnswer__Question--String.norm {}".format(args.APP, args.SUFFIX).strip()
	elif args.TYPE == "getDefinition":
		file_name = "{}/temp.txt.VpaGetDefinition__Question--String.norm {}".format(args.APP, args.SUFFIX).strip()
	else:
		file_name = "{}/temp-train.txt.norm {}".format(args.APP, args.SUFFIX).strip()
	# file_name = "{}/temp-train.txt.norm 5".format(APP)
	
	norm_dict = defaultdict(list)
	collect_features = defaultdict(int)
	with open(file_name, 'r') as filename:
		for line in filename:
			line_list = line.strip().split("\t")
			if len(line_list) < 3:
				continue
			for feat in line_list[2].split():
				collect_features[feat] += 1
			
			if COMPARE is not None and line_list[1] not in COMPARE:
				continue
			norm_dict[line_list[1]].append((line_list[0], line_list[2]))
	
	# count_PE = Counter([y for x in norm_dict["VpaDynamicallyCustomizable.PaymentExecution"] for y in x[1].split()])
	# count_PF = Counter([y for x in norm_dict["VpaDynamicallyCustomizable.PaymentFees"] for y in x[1].split()])
	print("check")
	K = int(args.K)
	doccs = []
	labs = []
	token_grams = []
	for label, values in norm_dict.items():
		labs.append(label)
		doccs.append(" ".join([x[1] for x in values]))
		this_doc = []
		for _id, val in values:
			token_list = make_replacements(val).split()
			if len(token_list)  < 1:
				continue
			t_gram = TokenGram(token_list, K)
			this_doc.append(t_gram.get_token())
		token_grams.append(" ".join(this_doc))
			
	# tfidf(docs=doccs, labels=labs, top_n=30)
	results = tfidf(docs=token_grams, labels=labs, n_gram = (1, int(args.N_GRAM)), top_n=int(args.TOP_N))
	with open("results/{}_{}_{}_{}_k={}.txt".format(args.APP, args.TYPE, args.OUT, args.SUFFIX, args.K), "w") as output_file_name:
		for label, result in results:
			print(label)
			print(result)
			print("\n")
			output_file_name.write("{}\n{}\n\n".format(label, result))


	print("finished")
