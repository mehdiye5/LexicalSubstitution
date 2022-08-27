import os, sys, optparse
import tqdm
import re
import numpy as np
import pymagnitude

# norm_word() and read_lexicon() are from retrofit.py
# -- https://github.com/mfaruqui/retrofitting/blob/master/retrofit.py -- 

isNumber = re.compile(r'\d+.*')
def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()

def read_lexicon(filename):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

# -- Our code from here -- 

def retrofit_semantic_lexicons(wv, word, ontology, dim, vector):
    """
    Function to update the word vectors using ontology weights.
    """
    # If the word is not found in ontology return original vector
    if word not in ontology or len(ontology[word]) == 0:
        return (vector, 0)
    else:
        # Tunable beta param 
        beta = 0.975 / len(ontology[word])
        sum1 = np.zeros(dim)
     
        # Getting numerator summation (from pseudocode)
        for similar_word in ontology[word]:
            if norm_word(similar_word) != '---num---' and norm_word(similar_word) != '---punc---' and similar_word in wv.keys():
                sum1 += beta * wv[similar_word]

        # Getting denominator summation - equal to performing summation inside loop
        sum2 = beta * len(ontology[word])
        return (sum1, sum2)


def main(opts):
    # Creating copy of word vector dict
    wv = pymagnitude.Magnitude(opts.wordvecfile)
    new_wv = {}

    for key, vector in tqdm.tqdm(wv, desc="Ontology"):
        new_wv[key] = vector

    # Loading Ontology 
    ontology = read_lexicon(opts.lexicon)
    alpha = 0.11

    # Iteratively updating word vectors
    for t in tqdm.tqdm(range(10), desc="Retrofitting"):
        for key, vector in wv:		
            sum1, sum2 = retrofit_semantic_lexicons(new_wv, key, ontology, wv.dim, vector)
            # If there are no edges in ontology graph, dont update embedding
            if sum2 != 0:
                new_wv[key] = (sum1 + alpha*vector) / (sum2 + alpha)

    # Writing new word embeddings to file
    with open(opts.output, 'w', encoding = 'utf-8') as outfile:
        for key, vector in new_wv.items():
            outfile.write(str(key) + " ")
            outfile.write(" ".join(map(str, vector)))
            outfile.write('\n')

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-w", "--wvfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="references word vector file path")
    optparser.add_option("-l", "--lexfile", dest="lexicon", default=os.path.join('data', 'lexicons', 'wordnet-synonyms.txt'), help="references lexicon file path")
    optparser.add_option("-o", "--outfile", dest="output", default=os.path.join('data', 'glove.6B.100d.retrofit.txt'), help="references output txt file path")
    (opts, _) = optparser.parse_args()

    main(opts)