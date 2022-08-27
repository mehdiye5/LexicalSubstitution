import os, sys, optparse, logging, heapq
import tqdm
import pymagnitude
import numpy as np

class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

    # def cosine_sim(a,b):
    #     return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def context_base_similarity(self, index, sentence):
        "Context based similarity implementation"

        # Get 4 surrounging context words
        context_words = []
        context_embedding = np.zeros(self.wvecs.dim)
        for idx in (index-1, index-2, index+1, index+2):
            if idx >= 0 and idx < len(sentence):
                context_words.append(sentence[idx])

        if len(context_words) != 0:
            for cword in context_words:
                context_embedding += self.wvecs.query(cword)
            context_embedding/=len(context_embedding)
        else:
            return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))


        # Get new embedding
        alpha = 0.01
        embedding = np.linalg.norm(self.wvecs.query(sentence[idx]) + (context_embedding*alpha))
        heap = []
        
        # Compute similarity across entire vocab
        for key, vector in self.wvecs:
            cs = self.wvecs.similarity(vector, embedding)
            if len(heap) < 10:
                heapq.heappush(heap, (cs, key))
            else:
                heapq.heappushpop(heap, (cs, key))
        
        return [x[1] for x in heap]

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.retrofit.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexsub = LexSub(opts.wordvecfile, int(opts.topn))
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            # print(" ".join(lexsub.context_base_similarity(int(fields[0].strip()), fields[1].strip().split())))
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
