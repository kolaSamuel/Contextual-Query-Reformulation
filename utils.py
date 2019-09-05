import string
import spacy
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from collections import defaultdict
import sys
from IPython.display import display
from random import randint
from copy import deepcopy
from functools import reduce
tqdm.pandas()
nlp = spacy.load('en_core_web_sm')


def slotExtractor(turn):
    """
    Extracts all unique slots from a sentence
    :param turn: string
    :return: list of all unique slots
    """
    results = []
    for token in nlp(turn):
        text = token.text
        if len(text) > 1 and text[-2] == '_':
            results.append(text)
    return results


def slotgenerator(slot):
    if type(slot[1]) != list:
        yield (slot[0], slot[1].strip().lower())
    else:
        for y in slot[1]:
            for x in slotgenerator((slot[0], y)):
                yield x
                
                
def dialogs2TurnInput(inputs, f=lambda x: x, turn=True, system=True, index=True):
    """
    Breaks down a dialog into subsets, each ending with the users turn
    :param inputs: dialog
    :param f: feature function to apply to x
    :param turn (boolean): use turn information
    :param system (boolean) : if system present in dialog or not
    :return: list subsets of dialog
    """
    results = []
    speaker = ('<user> ', '<system> ', ' 1@$% ')
    if not index:
        inputs, slots = inputs
    for i in range(0, len(inputs), 1+system):
        dialog = inputs[:i+1]
        if not index:
            rolebased(dialog, slots, turn_order=-1)
        text = " ".join([speaker[i%(1+system) if turn else 2]+f(x) for i, x in enumerate(dialog)])
        results.append(text)
    return results


def dialogs2TurnOutput(outputs, system=True, index=True):
    """
    Extracts user turns from dialog
    :param outputs (list): dialog
    :param system (boolean) : if system present in dialog or not
    :return: lists of user turns (matching the last turn produced by dialogs2Turn)
    """
    results = []
    if not index:
        outputs, slots = outputs
    for i in range(0, len(outputs), 1+system):
        turn = outputs[i:i+1]
        if not index:
            rolebased(turn, slots, turn_order=-1)
        string = "".join(turn[0])
        results.append(string)
    return results


def dialog2TurnInputEval(inputs, maxD=3):
    results = []
    for i in range(0, len(inputs), 2):
        d_data = defaultdict(set)
        for j in range(i, -1, -1):
            k = i-j
            d_data[min(k, maxD)].update(slotExtractor(inputs[j]))
        results.append(d_data)
    return results


def dialog2TurnOutputEval(inputs, maxD=3):
    results = []
    for i in range(0, len(inputs), 2):
        entities = set((slotExtractor(inputs[i])))
        results.append(entities)
    return results


def canonicalEnt(turn, allCQRFeatures=True):
    """
    Adds Canonical features (+ all other CQR features) to entities
    :param turn: turn in a dialog
    :param allCQRFeatures: Boolean to indicate if to add Possesibles, possesive pronouns and question tags
    :return: turn with canonical entites
    """
    results = []
    for token in nlp(turn):
        try:
            if token.text[-2] == '_':
                results.append(token.dep_)
            elif allCQRFeatures:
                raise IndexError
        except IndexError:
            if token.dep_ == 'poss':
                results.append('PSBL')
            elif token.pos_ == 'PRON':
                results.append('PRP$')
            elif token.text in ['what', 'when', 'where', 'who', 'which', 
                                'whom','whose', 'why', 'how' ]:
                results.append('QUESTION')
                
        results.append(token.text)
        
    return rejoin(results)

def rejoin(array):
    """
    Recontructs sentence using predefinded rules
    
    Example:
    >>>Token_senetence = ['the', 'boy', 'jumped', 'over', 'the', 'mary', "'s", 'house', ',', 'did', "n't", 'he', '?']
    >>>rejoin(Token_sentence)
    "the boy jumped over the mary's house, did n't he?"
    
    :param array (list): tokenized array of strings in a sentence 
    :return: sentence string 
    """
    to_Join = []
    for x in array:
        if x == '':
            continue
        elif x[0] in string.punctuation:
            to_Join.append(x)
        else:
            to_Join.append(' '+x)
    return "".join(to_Join).strip()

def replaceSlots(doc, slot, entities, del_unused=False, ordered=False, span=2):
    """
    Replaces slots with indexed entities
    Example:
    >>>doc = [will, it, be, humid, on, friday, and, saturday, in, new, york, ?]
    >>>slot = 'date'
    >>>entities = ['friday', 'saturday']
    >>>replaceSlots(doc, slot, entities)
    >>>doc
    [will, it, be, humid, on, date_1, and, date_2, in, new, york, ?]
    
    :param slot (string): slot type 
    :param entities (list): dictionary of entites
    :param del_unused (boolean): flag to inticate if unused entities should be removed or not
    :param ordered: nool indicating if to maintain current slot ordering during index
    :param span (integer): width to expand entity by if not found
    :return: array of text with indexed slots
    """
    i = 1
    used = False
    docCopy = doc[:]
    used_arr = []
    for entity in entities:
        if type(entity) != str:
            #this means that it an unsued entity and is of type tuple: (False, entity)
            continue
        for k in range(span):
            entity_arr = set([x.text for x in nlp(entity)]+['z^@']*k)
            l = len(entity_arr)
            for j in range(len(doc)-l+1):
                if len(set(docCopy[j:j+l]).intersection(entity_arr)) == l-k:
                    used = True
                    docCopy[j:j+l] = ''
                    doc[j] = slot + "_" + str(i)
                    doc[j+1:j+l] = ''
        i += used
        used_arr.append(used)
        used = False
        
    if del_unused:
        for i, isused in enumerate(used_arr):
            if not isused:
                entities[i] = (False, entities[i])

def rolebased(inputs, slots, del_unused=False, inplace=False, ordered=False, slot_type=True, turn_order=1):
    """
    Does role based entity indexing on string
    :param inputs (array): array of texts
    :param slots (dictionary): dictionary of slots
    :param del_unused (boolean): flag to inticate if unused entities should be removed or not
    :param inplace (boolean): bool for return
    :param ordered (boolean): nool indicating if to maintain current slot ordering during index
    :return: array of texts replaced with entity indexes 
    """
    sep = " 1@$% "
    #to low or not to low this is the question
    temps = np.array([x.text.lower() for x in nlp(sep.join(inputs))], dtype=np.object)
#     print(temps)
    if slot_type:
        for slot, entities in sorted(slots.items(), key=lambda x: x*(not ordered)):
            replaceSlots(temps, slot, entities[::turn_order], del_unused, ordered)
    else:
        single_slot = reduce(lambda y, x: y+x, 
                             [[entity for entity in entities] 
                              for slot, entities in sorted(slots.items())], 
                             []) 
        replaceSlots(temps, 'entity', single_slot, ordered)
    results = rejoin(temps).split(sep.strip())
    for i in range(len(results)):
        inputs[i] = results[i].strip()
    if not inplace:
        return inputs
        