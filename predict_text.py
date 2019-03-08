from collections import Counter
import numpy as np

from data_generator import AudioGenerator
from ASR_model import ASR_network


#################### TEXT PREDICTION W/ 2 ADDED FEATURES: #####################
############################## 1. SPELL CORRECTION ############################
corpus = AudioGenerator()
corpus.load_train_data("train_corpus.json")
corpus_texts = corpus.train_texts
corpus_tokens = np.concatenate(np.array([sentence.lower().split() for sentence in corpus_texts]))
token_counter = Counter(corpus_tokens)

# Spell checker by Peter Norvig (http://norvig.com/spell-correct.html)
# + my addition of vowel_replaces & conso_replaces for edits_1 function
def word_probability(word):
    ''' Return unigram word probabilty using Train Corpus '''
    
    return token_counter[word]/sum(token_counter.values())



def edits_1(word):
    ''' Performs ONE of deletion, transposition, replacement, or insertion to the given word '''
    
    letters = "abcdefghijklmnopqrstuvwxyz"
    vowels = "aeiouy"
    consonants = "bcdfghjklmnpqrstvwxz"
    splits = [(word[1:i], word[i:]) for i in range(len(word) + 1)]
    
    deletes    = [word[0] + L + R[1:]               for L, R in splits if R]
    transposes = [word[0] + L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    vowel_replaces = [word[0] + L + v + R[1:]       for L, R in splits if (R and R[0] in vowels) for v in vowels]        
    conso_replaces = [word[0] + L + c + R[1:]       for L, R in splits if (R and R[0] in consonants) for c in consonants]                                                              
    inserts    = [word[0] + L + ch + R              for L, R in splits for ch in letters]

    return set(deletes + transposes + vowel_replaces + conso_replaces + inserts)



def edits_2(word):
    ''' Performs TWO of deletion, transposition, replacement, or insertion to the given word '''
    
    edits_1_words = edits_1(word)
    edits_2 = set()
    for e1 in edits_1_words:
        for e2 in edits_1(e1):
            edits_2.add(e2)
    
    return edits_2



def existing_words(editted_words):
    ''' Returns a subset of editted words that exist in train corpus '''
    
    return set(w for w in editted_words if w in token_counter)



####################### 2. PART OF SPEECH (POS) TAGGING #######################
import nltk
nltk.download('averaged_perceptron_tagger')

from nltk import bigrams
from collections import Counter, defaultdict

# Create a bigram model for POS tags, base on Training Corpus
bigram_tags = defaultdict(lambda: defaultdict(lambda: 0))

# (1) Record bigram counts in a dict
for sentence in corpus_texts:
    tags = [nltk.pos_tag((w.split()))[0][1] for w in sentence.split()]
    for t1, t2 in bigrams(tags):
        bigram_tags[t1][t2] += 1
        
# (2) Transform the counts into probabilities
for t1 in bigram_tags:
    total_count = float(sum(bigram_tags[t1].values()))
    for t2 in bigram_tags[t1]:
        bigram_tags[t1][t2] /= total_count
        
        
        
def candidates_POS(word, prev_tag = None):
    ''' Returns possible spelling corrections for the given word '''
    
    if existing_words([word]):  # if predicted word appears in the corpus
        if prev_tag is None:
            return [word]
        else:
            tag = nltk.pos_tag((word.split()))[0][1]
            # if the word has a reasonable tag, considering previous 2 words' tags
            if bigram_tags[prev_tag][tag] > 0.01:  
                return [word]
            
    one_edits = existing_words(edits_1(word))
    if one_edits:   
        if prev_tag is None:
            return one_edits
        else:
            tags = [(w, nltk.pos_tag((w.split()))[0][1]) for w in one_edits]
            bigrams = [(w, bigram_tags[prev_tag][tag]) for (w, tag) in tags]
            logical_candidates = [pair[0] for pair in bigrams if pair[1] > 0.01]
            if logical_candidates:
                return logical_candidates

    two_edits = existing_words(edits_2(word))
    if two_edits:
        if prev_tag is None:
            return two_edits
        else:
            tags = [(w, nltk.pos_tag((w.split()))[0][1]) for w in two_edits]
            bigrams = [(w, bigram_tags[prev_tag][tag]) for (w, tag) in tags]
            logical_candidates = [pair[0] for pair in bigrams if pair[1] > 0.01]
            if logical_candidates:
                return logical_candidates
    
    return [word]   

        

def correction_POS(words):
    ''' Returns the most probabale spelling correction of the given sentence (words)'''
    
    corrected_sentence = []    
    for word_i, word in enumerate(words): 
        # Don't apply POS tagging selection to the first word
        if word_i == 0:
            prev_tag = None    

        word_candidates = candidates_POS(word, prev_tag = prev_tag)
        next_word = max(word_candidates, key = word_probability)
        corrected_sentence.append(next_word) 
        prev_tag = nltk.pos_tag((next_word.split()))[0][1]
            
    return corrected_sentence



############################## MAKE PREDICTIONS! ##############################
from keras import backend as K
from utils import int_sequence_to_text

def get_predictions(index, 
                    partition, 
                    ASR_model = ASR_network, 
                    model_path = 'results/model_final_43_epochs.h5'):
    """ Prints a model's decoded predictions
    Params:
        index (int): sample index of training or validation set
        partition (str): One of 'train' or 'validation'
        ASR_model (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # 1. Load the train and validation data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    
    # 2. Obtain the true transcription and the audio features 
    if partition == 'train':
        true_label = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]              
        
    elif partition == 'validation':
        true_label = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]      
    
    data_point = data_gen.normalize(data_gen.featurize(audio_path)) 
    
    # 3. Obtain and decode the acoustic model's predictions
    ASR_model.load_weights(model_path)
    prediction = ASR_model.predict(np.expand_dims(data_point, axis=0))  # give a batch size of 1
    output_length = [ASR_model.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(prediction, output_length)[0][0]) + 1).flatten().tolist()
    pred_words = ''.join(int_sequence_to_text(pred_ints)).split()
    
    # 4. Perform Spelling Correction & POS Tagging
    corrected_str_POS = correction_POS(pred_words)
    
    # 5. play the audio file, and display the true and predicted transcriptions
    print('-'*80)
    print('True transcription:\n' + '\n' + true_label)
    print('-'*80)
    print('Predicted transcription: (Original -- Spell Correction, POS tagging)\n')
    print(' '.join(pred_words))
    print(' '.join(corrected_str_POS))
    print('-'*80)
    
    
    
get_predictions(index = 0, partition = 'train')
get_predictions(index = 1000, partition = 'train')
get_predictions(index = 999, partition = 'validation')