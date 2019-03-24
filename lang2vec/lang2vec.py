#!/usr/bin/env python

from __future__ import print_function
from __future__ import unicode_literals

import argparse, json, itertools, os
import numpy as np
import pkg_resources

''' 
Turning the convenience script into a library, for accessing the values inside the URIEL typological and geodata knowledge bases 

Author: Antonis Anastasopoulos
Original Author: Patrick Littell
Last modified: July 15, 2016
'''

LETTER_CODES_FILE = pkg_resources.resource_filename(__name__, "data/letter_codes.json")
FEATURE_SETS_DICT = {
    
    "syntax_wals" : ( "features.npz", "WALS", "S_" ),
    "phonology_wals": ( "features.npz", "WALS", "P_" ),
    "syntax_sswl" : ( "features.npz", "SSWL", "S_" ),
    "syntax_ethnologue": ( "features.npz", "ETHNO", "S_" ),
    "phonology_ethnologue" : ( "features.npz", "ETHNO", "P_" ),
    "inventory_ethnologue" : ( "features.npz", "ETHNO", "INV_" ),
    "inventory_phoible_aa" : ( "features.npz", "PHOIBLE_AA", "INV_" ),
    "inventory_phoible_gm" : ( "features.npz", "PHOIBLE_GM", "INV_" ),
    "inventory_phoible_saphon" : ( "features.npz", "PHOIBLE_SAPHON", "INV_"),
    "inventory_phoible_spa" : ( "features.npz", "PHOIBLE_SPA", "INV_" ),
    "inventory_phoible_ph" : ( "features.npz", "PHOIBLE_PH", "INV_" ),
    "inventory_phoible_ra" : ( "features.npz", "PHOIBLE_RA", "INV_" ),
    "inventory_phoible_upsid" : ( "features.npz", "PHOIBLE_UPSID", "INV_" ),
    "syntax_knn" : ( "feature_predictions.npz", "predicted", "S_" ),
    "phonology_knn" : ( "feature_predictions.npz", "predicted", "P_" ),
    "inventory_knn" : ( "feature_predictions.npz", "predicted", "INV_" ),
    "syntax_average" : ( "feature_averages.npz", "avg", "S_" ),
    "phonology_average" : ( "feature_averages.npz", "avg", "P_" ),
    "inventory_average" : ( "feature_averages.npz", "avg", "INV_" ),
    "fam" : ( "family_features.npz", "FAM", ""),
    "geo" : ( "geocoord_features.npz", "GEOCOORDS", ""),
    
}


#LETTER_CODES = {"am": "amh", "bs": "bos", "vi": "vie", "wa": "wln", "eu": "eus", "so": "som", "el": "ell", "aa": "aar", "or": "ori", "sm": "smo", "gn": "grn", "mi": "mri", "pi": "pli", "ps": "pus", "ms": "msa", "sa": "san", "ko": "kor", "sd": "snd", "hz": "her", "ks": "kas", "fo": "fao", "iu": "iku", "tg": "tgk", "dz": "dzo", "ar": "ara", "fa": "fas", "es": "spa", "my": "mya", "mg": "mlg", "st": "sot", "gu": "guj", "uk": "ukr", "lv": "lav", "to": "ton", "nv": "nav", "kl": "kal", "ka": "kat", "yi": "yid", "pl": "pol", "ht": "hat", "lu": "lub", "fr": "fra", "ia": "ina", "lt": "lit", "om": "orm", "qu": "que", "no": "nor", "sr": "srp", "br": "bre", "rm": "roh", "io": "ido", "gl": "glg", "nb": "nob", "ng": "ndo", "ts": "tso", "nr": "nbl", "ee": "ewe", "bo": "bod", "mt": "mlt", "ta": "tam", "et": "est", "yo": "yor", "tw": "twi", "sl": "slv", "su": "sun", "gv": "glv", "lo": "lao", "af": "afr", "sg": "sag", "sv": "swe", "ne": "nep", "ie": "ile", "bm": "bam", "sc": "srd", "sw": "swa", "nn": "nno", "ho": "hmo", "ak": "aka", "ab": "abk", "ti": "tir", "fy": "fry", "cr": "cre", "sh": "hbs", "ny": "nya", "uz": "uzb", "as": "asm", "ky": "kir", "av": "ava", "ig": "ibo", "zh": "zho", "tr": "tur", "hu": "hun", "pt": "por", "fj": "fij", "hr": "hrv", "it": "ita", "te": "tel", "rw": "kin", "kk": "kaz", "hy": "hye", "wo": "wol", "jv": "jav", "oc": "oci", "kn": "kan", "cu": "chu", "ln": "lin", "ha": "hau", "ru": "rus", "pa": "pan", "cv": "chv", "ss": "ssw", "ki": "kik", "ga": "gle", "dv": "div", "vo": "vol", "lb": "ltz", "ce": "che", "oj": "oji", "th": "tha", "ff": "ful", "kv": "kom", "tk": "tuk", "kr": "kau", "bg": "bul", "tt": "tat", "ml": "mal", "tl": "tgl", "mr": "mar", "hi": "hin", "ku": "kur", "na": "nau", "li": "lim", "nl": "nld", "nd": "nde", "os": "oss", "la": "lat", "bn": "ben", "kw": "cor", "id": "ind", "ay": "aym", "xh": "xho", "zu": "zul", "cs": "ces", "sn": "sna", "de": "deu", "co": "cos", "sk": "slk", "ug": "uig", "rn": "run", "he": "heb", "ba": "bak", "ro": "ron", "be": "bel", "ca": "cat", "kj": "kua", "ja": "jpn", "ch": "cha", "ik": "ipk", "bi": "bis", "an": "arg", "cy": "cym", "tn": "tsn", "mk": "mkd", "ve": "ven", "eo": "epo", "kg": "kon", "km": "khm", "se": "sme", "ii": "iii", "az": "aze", "en": "eng", "ur": "urd", "za": "zha", "is": "isl", "mh": "mah", "mn": "mon", "sq": "sqi", "lg": "lug", "gd": "gla", "fi": "fin", "ty": "tah", "da": "dan", "si": "sin", "ae": "ave"}
with open(LETTER_CODES_FILE, 'r') as letter_file:
    LETTER_CODES = json.load(letter_file)

def available_languages():
    return set([LETTER_CODES[lang_code] for lang_code in LETTER_CODES])

def available_feature_sets():
    return [key for key in FEATURE_SETS_DICT]

LANGUAGES = available_languages()
FEATURE_SETS = available_feature_sets()

def get_language_code(lang_code, feature_database):
    # first, normalize to an ISO 639-3 code
    if lang_code in LETTER_CODES:
        lang_code = LETTER_CODES[lang_code]
    if lang_code not in feature_database["langs"]:
        #print("ERROR: Language " + lang_code + " not found.", file=sys.stderr)
        raise Exception("ERROR: Language " + lang_code + " not found. " +
            "Run lang2vec.LANGUAGES or lang2vec.available_languages()" + 
            " to see a list of supported languages.")
    return lang_code

def get_language_index(lang_code, feature_database):
    return np.where(feature_database["langs"] == lang_code)[0][0]

def get_source_index(source_name, feature_database):
    return np.where(feature_database["sources"] == source_name)[0]

def get_feature_names(feature_name_prefix, feature_database):
    return [ f for f in feature_database["feats"] if f.startswith(feature_name_prefix) ]

def get_feature_index(feature_name, feature_database):
    return np.where(feature_database["feats"] == feature_name)[0][0]
    
def get_id_set(lang_codes):
    feature_database = np.load("family_features.npz")
    lang_codes = [ get_language_code(l, feature_database) for l in lang_codes ]
    all_languages = list(feature_database["langs"])
    feature_names = [ "ID_" + l.upper() for l in all_languages ]
    values = np.zeros((len(lang_codes), len(feature_names)))
    for i, lang_code in enumerate(lang_codes):
        feature_index = get_language_index(lang_code, feature_database)
        values[i, feature_index] = 1.0
    return feature_names, values
    
    
def get_named_set(lang_codes, feature_set):
    if feature_set == 'id':
        return get_id_set(lang_codes)
    
    if feature_set not in FEATURE_SETS_DICT:
        raise Exception("ERROR: Invalid feature set " + feature_set +
            ". You can run lang2vec.FEATURE_SETS or " + 
            " lang2vec.available_feature_sets() to see the available feature sets.")
        
    filename, source, prefix = FEATURE_SETS_DICT[feature_set]
    filename = pkg_resources.resource_filename(__name__, os.path.join('data', filename))
    feature_database = np.load(filename)
    lang_codes = [ get_language_code(l, feature_database) for l in lang_codes ]
    lang_indices = [ get_language_index(l, feature_database) for l in lang_codes ]
    feature_names = get_feature_names(prefix, feature_database)
    feature_indices = [ get_feature_index(f, feature_database) for f in feature_names ]
    source_index = get_source_index(source, feature_database)
    feature_values = feature_database["data"][lang_indices,:,:][:,feature_indices,:][:,:,source_index]
    feature_values = feature_values.squeeze(axis=2)
    return feature_names, feature_values

def get_union_sets(lang_codes, feature_set_str):
    if isinstance(feature_set_str, str):
        feature_set_parts = feature_set_str.split("|")
    elif isinstance(feature_set_str, list):
        feature_set_parts = feature_set_str
    else:
        raise Exception("Improper type "+type(feature_set_str)+" for feature_set.\nRequires string or list of strings.")
    feature_names, feature_values = get_named_set(lang_codes, feature_set_parts[0])
    for feature_set_part in feature_set_parts[1:]:
        more_feature_names, more_feature_values = get_named_set(lang_codes, feature_set_part)
        if len(feature_names) != len(more_feature_names):
            #print("ERROR: Cannot perform elementwise union on feature sets of different size")
            raise Exception("ERROR: Cannot perform elementwise union on feature sets of different size")
            #sys.exit(0)
        feature_values = np.maximum(feature_values, more_feature_values)
    return feature_names, feature_values
    
def get_concatenated_sets(lang_codes, feature_set_str):
    if isinstance(feature_set_str, str):
        feature_set_parts = feature_set_str.split("+")
    elif isinstance(feature_set_str, list):
        feature_set_parts = feature_set_str
    else:
        raise Exception("Improper type "+type(feature_set_str)+" for feature_set.\nRequires string or list of strings.")
    feature_names = []
    feature_values = np.ndarray((len(lang_codes),0))
    for feature_set_part in feature_set_parts:
        more_feature_names, more_feature_values = get_union_sets(lang_codes, feature_set_part)
        feature_names += more_feature_names
        feature_values = np.concatenate([feature_values, more_feature_values], axis=1)
    return feature_names, feature_values

def fs_concatenation(fs1, *args):
    fs_s = []
    if isinstance(fs1, str):
        if fs1 in FEATURE_SETS_DICT or '|' in fs1:
            fs_s.append(fs1)
        else:
            raise Exception("ERROR: Invalid feature set " + fs1 +
                ".\nYou can run lang2vec.FEATURE_SETS or " + 
                " lang2vec.available_feature_sets() to see the available feature sets.")
    elif isinstance(fs1, list):
        for fs in fs1:
            if fs in FEATURE_SETS_DICT or '|' in fs:
                fs_s.append(fs)
            else:
                raise Exception("ERROR: Invalid feature set " + fs +
                    ".\nYou can run lang2vec.FEATURE_SETS or " + 
                    " lang2vec.available_feature_sets() to see the available feature sets.")
    else:
        raise Exception("ERROR: Invalid feature set type " + type(fs1) + " (needs string or list).")

    if args:
        for fs in args:
            if isinstance(fs, str):
                if '|' in fs or fs in FEATURE_SETS_DICT:
                    fs_s.append(fs)
                else:
                    raise Exception("ERROR: Invalid feature set " + fs +
                        ".\nYou can run lang2vec.FEATURE_SETS or " + 
                        " lang2vec.available_feature_sets() to see the available feature sets.")
            elif isinstance(fs, list):
                for fs_t in fs:
                    if '|' in fs_t or fs_t in FEATURE_SETS_DICT:
                        fs_s.append(fs_t)
                    else:
                        raise Exception("ERROR: Invalid feature set " + fs_t +
                            ".\nYou can run lang2vec.FEATURE_SETS or " + 
                            " lang2vec.available_feature_sets() to see the available feature sets.")
    return '+'.join(fs_s)


def fs_union(fs1,*args):
    fs_s = []
    if isinstance(fs1, str):
        if fs1 not in FEATURE_SETS_DICT:
            raise Exception("ERROR: Invalid feature set " + fs1 +
                ".\nYou can run lang2vec.FEATURE_SETS or " + 
                " lang2vec.available_feature_sets() to see the available feature sets.")
        else:
            fs_s.append(fs1)
    elif isinstance(fs1, list):
        for fs in fs1:
            if fs not in FEATURE_SETS_DICT:
                raise Exception("ERROR: Invalid feature set " + fs +
                    ".\nYou can run lang2vec.FEATURE_SETS or " + 
                    " lang2vec.available_feature_sets() to see the available feature sets.")
            else:
                fs_s.append(fs)
    else:
        raise Exception("ERROR: Invalid feature set type " + type(fs1) + " (needs string or list).")

    if args:
        for fs in args:
            if isinstance(fs, str):
                if fs not in FEATURE_SETS_DICT:
                    raise Exception("ERROR: Invalid feature set " + fs +
                        ".\nYou can run lang2vec.FEATURE_SETS or " + 
                        " lang2vec.available_feature_sets() to see the available feature sets.")
                else:
                    fs_s.append(fs)
            elif isinstance(fs, list):
                for fs_t in fs:
                    if fs_t not in FEATURE_SETS_DICT:
                        raise Exception("ERROR: Invalid feature set " + fs_t +
                            ".\nYou can run lang2vec.FEATURE_SETS or " + 
                            " lang2vec.available_feature_sets() to see the available feature sets.")
                    else:
                        fs_s.append(fs_t)
    return '|'.join(fs_s)
    
    

def get_features(languages, feature_set_inp, header=False, minimal=False):    
    if isinstance(languages, str):
        lang_codes = languages.split()
    elif isinstance(languages, list):
        lang_codes = languages 
    else:
        raise Exception("Improper type "+type(languages)+" for languages.\nRequires string or list of strings.")
        
    feature_names, feature_values = get_concatenated_sets(lang_codes, feature_set_inp)
    feature_names = np.array([ f.replace(" ","_") for f in feature_names ])

    if minimal:
        mask = np.all(feature_values == 0.0, axis=0)
        mask |= np.all(feature_values == 1.0, axis=0)
        mask |= np.all(feature_values == -1.0, axis=0)
        unmasked_indices = np.where(np.logical_not(mask))
    else:
        unmasked_indices = np.where(np.ones(feature_values.shape[1]))
    
    output = {}
    if header:
        output['CODE']=list(feature_names[unmasked_indices])
        
    for i, lang_code in enumerate(lang_codes):
        values = feature_values[i,unmasked_indices].ravel()
        values = [ '--' if f == -1 else f for f in values ]
        #print("\t".join([lang_code]+values))
        output[lang_code] = values
    return output

'''
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("languages", default='', help="The languages of interest, in ISO 639-3 codes, separated by spaces (e.g., \"deu eng fra swe\")")
    argparser.add_argument("feature_set", default='', help="The feature set or sets of interest (e.g., \"syntax_knn\" or \"fam\"), joined by concatenation (+) or element-wise union (|).")
    argparser.add_argument("-f", "--fields", default=False, action="store_true", help="Print feature names as the first row of data.")
    argparser.add_argument("-r", "--random", default=False, action="store_true", help="Randomize all feature values (e.g., to make a control group).")
    argparser.add_argument("-m", "--minimal", default=False, action="store_true", help="Suppress columns that are all 0, all 1, or all nulls.")
    args = argparser.parse_args()
    get_features(args.languages, args.feature_set, args.fields, args.random, args.minimal)
'''