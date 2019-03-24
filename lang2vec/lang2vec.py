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
    "id" : ( "family_features.npz", "ID", ""),
    "geo" : ( "geocoord_features.npz", "GEOCOORDS", ""),
    "learned" : ( "learned.npy", "learned", "LEARNED_")
    
}


#LETTER_CODES = {"am": "amh", "bs": "bos", "vi": "vie", "wa": "wln", "eu": "eus", "so": "som", "el": "ell", "aa": "aar", "or": "ori", "sm": "smo", "gn": "grn", "mi": "mri", "pi": "pli", "ps": "pus", "ms": "msa", "sa": "san", "ko": "kor", "sd": "snd", "hz": "her", "ks": "kas", "fo": "fao", "iu": "iku", "tg": "tgk", "dz": "dzo", "ar": "ara", "fa": "fas", "es": "spa", "my": "mya", "mg": "mlg", "st": "sot", "gu": "guj", "uk": "ukr", "lv": "lav", "to": "ton", "nv": "nav", "kl": "kal", "ka": "kat", "yi": "yid", "pl": "pol", "ht": "hat", "lu": "lub", "fr": "fra", "ia": "ina", "lt": "lit", "om": "orm", "qu": "que", "no": "nor", "sr": "srp", "br": "bre", "rm": "roh", "io": "ido", "gl": "glg", "nb": "nob", "ng": "ndo", "ts": "tso", "nr": "nbl", "ee": "ewe", "bo": "bod", "mt": "mlt", "ta": "tam", "et": "est", "yo": "yor", "tw": "twi", "sl": "slv", "su": "sun", "gv": "glv", "lo": "lao", "af": "afr", "sg": "sag", "sv": "swe", "ne": "nep", "ie": "ile", "bm": "bam", "sc": "srd", "sw": "swa", "nn": "nno", "ho": "hmo", "ak": "aka", "ab": "abk", "ti": "tir", "fy": "fry", "cr": "cre", "sh": "hbs", "ny": "nya", "uz": "uzb", "as": "asm", "ky": "kir", "av": "ava", "ig": "ibo", "zh": "zho", "tr": "tur", "hu": "hun", "pt": "por", "fj": "fij", "hr": "hrv", "it": "ita", "te": "tel", "rw": "kin", "kk": "kaz", "hy": "hye", "wo": "wol", "jv": "jav", "oc": "oci", "kn": "kan", "cu": "chu", "ln": "lin", "ha": "hau", "ru": "rus", "pa": "pan", "cv": "chv", "ss": "ssw", "ki": "kik", "ga": "gle", "dv": "div", "vo": "vol", "lb": "ltz", "ce": "che", "oj": "oji", "th": "tha", "ff": "ful", "kv": "kom", "tk": "tuk", "kr": "kau", "bg": "bul", "tt": "tat", "ml": "mal", "tl": "tgl", "mr": "mar", "hi": "hin", "ku": "kur", "na": "nau", "li": "lim", "nl": "nld", "nd": "nde", "os": "oss", "la": "lat", "bn": "ben", "kw": "cor", "id": "ind", "ay": "aym", "xh": "xho", "zu": "zul", "cs": "ces", "sn": "sna", "de": "deu", "co": "cos", "sk": "slk", "ug": "uig", "rn": "run", "he": "heb", "ba": "bak", "ro": "ron", "be": "bel", "ca": "cat", "kj": "kua", "ja": "jpn", "ch": "cha", "ik": "ipk", "bi": "bis", "an": "arg", "cy": "cym", "tn": "tsn", "mk": "mkd", "ve": "ven", "eo": "epo", "kg": "kon", "km": "khm", "se": "sme", "ii": "iii", "az": "aze", "en": "eng", "ur": "urd", "za": "zha", "is": "isl", "mh": "mah", "mn": "mon", "sq": "sqi", "lg": "lug", "gd": "gla", "fi": "fin", "ty": "tah", "da": "dan", "si": "sin", "ae": "ave"}
with open(LETTER_CODES_FILE, 'r') as letter_file:
    LETTER_CODES = json.load(letter_file)

LEARNED_LETTER_CODES = {'nhy', 'wbp', 'dgc', 'aia', 'aim', 'aii', 'ztq', 'blw', 'dgr', 'blz', 'xav', 'dgz', 'kjh', 'mco', 'lhu', 'xnn', 'kjb', 'kje', 'lom', 'dhg', 'ibo', 'lhi', 'iba', 'zty', 'lac', 'tgp', 'rmy', 'lam', 'mxb', 'laj', 'ctu', 'lav', 'rmc', 'lat', 'rme', 'mxq', 'lug', 'cta', 'mxt', 'rmo', 'rmn', 'pag', 'kmu', 'ngp', 'ngu', 'kms', 'kmm', 'kmo', 'kmh', 'pwg', 'kmk', 'ngc', 'tha', 'tih', 'msc', 'plt', 'plu', 'bqc', 'plw', 'kxw', 'hla', 'ary', 'cjo', 'bqj', 'arb', 'nph', 'npo', 'npl', 'cjv', 'arn', 'arl', 'nya', 'qub', 'ksf', 'mjw', 'tmd', 'snd', 'nyn', 'ksr', 'kss', 'ksp', 'pes', 'cuc', 'mjc', 'ese', 'jam', 'wbm', 'maa', 'maf', 'sxb', 'mhx', 'maj', 'huv', 'huu', 'yua', 'hus', 'jac', 'mam', 'cme', 'maq', 'mav', 'mau', 'maz', 'jav', 'hub', 'yut', 'dhm', 'esk', 'omw', 'tur', 'bgs', 'tui', 'tuo', 'qul', 'tuc', 'quh', 'lww', 'tuf', 'tue', 'guo', 'aeu', 'wmt', 'aey', 'aeb', 'tbk', 'bhl', 'shn', 'tbo', 'tbl', 'bhg', 'wmw', 'tbg', 'sbl', 'rim', 'wiu', 'ziw', 'nbc', 'tke', 'nbe', 'slv', 'ria', 'knj', 'sba', 'ind', 'tku', 'inb', 'wib', 'sue', 'ino', 'wim', 'gbr', 'rkb', 'sgb', 'leu', 'kac', 'gur', 'due', 'leg', 'ium', 'dur', 'leh', 'sgz', 'kjs', 'msk', 'msm', 'msa', 'msb', 'snf', 'sna', 'hch', 'snc', 'tlf', 'sny', 'cco', 'nko', 'myw', 'snp', 'etu', 'krj', 'ntp', 'suk', 'cnt', 'mna', 'sua', 'suc', 'cnl', 'mpp', 'cni', 'cnh', 'suz', 'mse', 'fra', 'naf', 'sus', 'bzd', 'tpa', 'pap', 'qxo', 'qxn', 'bzh', 'qxh', 'bzj', 'njb', 'tpp', 'pab', 'jiv', 'tpt', 'kwi', 'gud', 'tpz', 'pah', 'pao', 'ong', 'qwh', 'ann', 'bci', 'bch', 'gnw', 'bcl', 'fuq', 'meq', 'guh', 'gnb', 'lai', 'gng', 'lol', 'anv', 'med', 'mee', 'dow', 'gid', 'zpo', 'zpl', 'zpm', 'eus', 'zpi', 'yby', 'ign', 'zpz', 'zpv', 'zpt', 'zpu', 'ons', 'giz', 'zpq', 'myk', 'ycn', 'cux', 'qxr', 'myb', 'for', 'mya', 'cul', 'myy', 'fon', 'cuk', 'klt', 'hwc', 'cub', 'myu', 'kbq', 'kbp', 'atg', 'rup', 'rus', 'urd', 'urb', 'ura', 'kbc', 'ton', 'toi', 'tgk', 'toj', 'kbh', 'fub', 'kbm', 'nfa', 'toc', 'tob', 'poy', 'iqw', 'bth', 'rro', 'nog', 'mpt', 'mpx', 'ken', 'por', 'yuj', 'pot', 'dyi', 'kek', 'poi', 'poh', 'btx', 'kew', 'loz', 'pol', 'nou', 'not', 'mph', 'awb', 'poe', 'kwd', 'hop', 'enx', 'hot', 'srq', 'srp', 'njz', 'sri', 'ixl', 'mww', 'srm', 'cgc', 'wuv', 'srn', 'yom', 'udu', 'urt', 'mbd', 'mbc', 'mbb', 'mbl', 'mbj', 'yva', 'mbh', 'amr', 'mbt', 'mbs', 'kmr', 'syb', 'hto', 'zaa', 'ukr', 'ttq', 'ivb', 'ojb', 'tte', 'ttc', 'crh', 'nmf', 'crm', 'boa', 'crn', 'bon', 'bom', 'boj', 'crx', 'aji', 'crs', 'crt', 'box', 'nak', 'kki', 'loq', 'aak', 'aai', 'kkc', 'nab', 'icr', 'dob', 'gub', 'guc', 'aaz', 'ood', 'jvn', 'naq', 'gui', 'nas', 'aau', 'gul', 'gum', 'gun', 'nav', 'ata', 'atb', 'pam', 'njm', 'njn', 'pri', 'prf', 'ivv', 'yor', 'zav', 'zas', 'zar', 'zam', 'zao', 'pls', 'zai', 'att', 'yon', 'zae', 'zad', 'sda', 'dtp', 'prs', 'zac', 'zab', 'mto', 'som', 'mti', 'xog', 'hbo', 'mta', 'des', 'uvl', 'sot', 'mtp', 'sop', 'soq', 'ckb', 'kyq', 'kyu', 'tyv', 'kyz', 'nst', 'nss', 'wnc', 'nso', 'nsn', 'asg', 'bpr', 'bps', 'kyf', 'kyg', 'nse', 'aso', 'ifu', 'nsa', 'kpg', 'kpf', 'mks', 'tso', 'tsn', 'eka', 'pdt', 'kpj', 'tsg', 'kpw', 'cax', 'tsz', 'mkd', 'kpr', 'pdc', 'mkl', 'mkn', 'kpx', 'xuo', 'bbj', 'dah', 'aoj', 'plg', 'bba', 'daa', 'zpc', 'nop', 'gof', 'mfe', 'mfk', 'txq', 'mua', 'txu', 'bbr', 'qvn', 'mos', 'zsr', 'gvl', 'hye', 'gvn', 'wnu', 'gvc', 'gvf', 'djr', 'bkq', 'afr', 'ksc', 'taw', 'tav', 'fai', 'tat', 'lcm', 'taq', 'tam', 'mek', 'taj', 'bkd', 'xla', 'tac', 'iou', 'gqr', 'kog', 'cak', 'cym', 'cya', 'snn', 'zat', 'mqy', 'kor', 'nep', 'zho', 'shu', 'nng', 'apy', 'nnb', 'apz', 'apu', 'apt', 'apw', 'nno', 'nnh', 'apr', 'mqb', 'adi', 'apn', 'nnp', 'yka', 'ape', 'tfr', 'kze', 'cpu', 'pne', 'hns', 'nwi', 'adj', 'ssw', 'wrk', 'cut', 'nwx', 'mhl', 'wrs', 'hne', 'hnj', 'ajz', 'mxp', 'ssg', 'ssd', 'gdn', 'ots', 'khy', 'otq', 'gdg', 'cot', 'fuv', 'cor', 'moc', 'con', 'ote', 'mox', 'fuh', 'cof', 'otn', 'otm', 'coe', 'mop', 'gdr', 'ncj', 'nxd', 'ktu', 'beq', 'tew', 'twi', 'mps', 'bem', 'bel', 'ben', 'ktj', 'tet', 'kto', 'dad', 'ktm', 'bef', 'csk', 'fij', 'cso', 'eza', 'fin', 'bnp', 'tsw', 'deu', 'xon', 'obo', 'csy', 'akh', 'nca', 'ake', 'niy', 'ded', 'khm', 'teo', 'rop', 'zos', 'tee', 'ted', 'zom', 'ilb', 'khz', 'grc', 'abt', 'ter', 'aby', 'abx', 'khs', 'ilo', 'byx', 'lmk', 'ike', 'sab', 'mai', 'yle', 'mzh', 'caf', 'shp', 'mza', 'ikk', 'lgm', 'kix', 'ikw', 'sat', 'mzz', 'roo', 'ron', 'xho', 'vie', 'vid', 'niq', 'usa', 'dww', 'ayr', 'hun', 'dwr', 'heg', 'tnp', 'tnn', 'tnk', 'tgl', 'nif', 'nii', 'kck', 'byr', 'tnc', 'nin', 'chf', 'che', 'chd', 'xtd', 'dan', 'bss', 'bsp', 'xtm', 'xtn', 'top', 'oym', 'tos', 'nrf', 'bsc', 'nri', 'ruf', 'chz', 'ell', 'bsn', 'mlp', 'kqp', 'trc', 'kqs', 'too', 'opm', 'kqw', 'mlg', 'swe', 'trp', 'kqc', 'mfy', 'swh', 'ppo', 'mcp', 'mcq', 'alb', 'ald', 'aom', 'mck', 'bbb', 'mcn', 'tod', 'alq', 'mca', 'mcb', 'mcd', 'mcf', 'okv', 'qup', 'khk', 'est', 'esu', 'quy', 'lus', 'quc', 'yim', 'quf', 'qug', 'mqj', 'soy', 'esi', 'bao', 'ban', 'gkp', 'kwj', 'lbb', 'awx', 'bjv', 'cwe', 'dis', 'pad', 'bjr', 'nob', 'lbj', 'lbk', 'sbe', 'noa', 'qvm', 'cag', 'yrb', 'dik', 'alp', 'tif', 'nde', 'yaq', 'ndo', 'tim', 'ndj', 'klv', 'ndi', 'yad', 'yre', 'yaa', 'yan', 'yao', 'yal', 'yam', 'tiy', 'upv', 'auc', 'nmo', 'iws', 'sey', 'gyr', 'kgf', 'aui', 'nma', 'usp', 'ses', 'mzl', 'seh', 'gya', 'gym', 'auy', 'vmy', 'kgp', 'cap', 'caq', 'car', 'kaq', 'cat', 'cav', 'slk', 'tzj', 'bjp', 'hau', 'sll', 'caa', 'cab', 'cac', 'gmv', 'mux', 'muy', 'wed', 'tna', 'hag', 'cao', 'geb', 'mpg', 'clu', 'ubu', 'ubr', 'atd', 'djk', 'rai', 'cle', 'xpe', 'kup', 'uig', 'mpm', 'tvk', 'kud', 'kue', 'bdh', 'kua', 'epo', 'kum', 'bdd', 'wer', 'bmh', 'bmk', 'ctd', 'yss', 'pis', 'gle', 'bmr', 'bmu', 'zaw', 'gla', 'mgc', 'cbi', 'zul', 'agm', 'agn', 'ian', 'vut', 'lit', 'agd', 'agg', 'gwi', 'mwm', 'lif', 'agr', 'lid', 'agt', 'agu', 'xbi', 'chq', 'ewe', 'azb', 'azg', 'yml', 'faa', 'azz', 'tbc', 'ptp', 'kqe', 'ptu', 'fao', 'nhr', 'sim', 'nhu', 'isd', 'ffm', 'nhx', 'cbr', 'cbu', 'cbt', 'cbv', 'rwo', 'vap', 'cbk', 'var', 'nhg', 'nhd', 'nhe', 'mwp', 'cbc', 'nhi', 'nho', 'kne', 'ebk', 'pmx', 'xsi', 'tcc', 'ceb', 'mrg', 'bvr', 'knf', 'mri', 'ces', 'miz', 'nvm', 'jic', 'pfe', 'mir', 'izz', 'miq', 'sps', 'mit', 'spp', 'mih', 'spl', 'mio', 'mil', 'mib', 'mic', 'sag', 'agw', 'mig', 'spa', 'mie', 'hrv', 'amk', 'amh', 'amn', 'amm', 'nbl', 'gaw', 'ame', 'gam', 'gah', 'hra', 'lsi', 'amp', 'amu', 'qvw', 'hix', 'qvs', 'waj', 'etr', 'qvz', 'ifa', 'ghs', 'qve', 'knv', 'qvc', 'ify', 'wat', 'zia', 'tzh', 'xed', 'wap', 'tzo', 'qvi', 'qvh', 'pma', 'rad', 'biu', 'tcs', 'adz', 'cpc', 'cpb', 'cpa', 'mbi', 'cpy', 'rar', 'tca', 'big', 'lmp', 'adl', 'kia', 'acf', 'ncl', 'aca', 'nch', 'acn', 'kik', 'ach', 'kin', 'mlh', 'jpn', 'acu', 'kqf', 'acr', 'whk', 'gso', 'ncu', 'nct', 'imo', 'kdi', 'kdc', 'buk', 'bum', 'bul', 'kde', 'nld', 'kdh', 'cui', 'kdl', 'sja', 'ita', 'zca', 'avt', 'mmx', 'pps', 'duo', 'ipi', 'mva', 'pio', 'pib', 'smo', 'mvn', 'smk', 'ven', 'cfm', 'pir', 'jra', 'gfk', 'lex', 'xsb', 'xsm', 'bru', 'nuy', 'hin', 'hil', 'ctp', 'xsr', 'bre', 'stp', 'nhw', 'emi', 'njo', 'viv', 'kvn', 'nzm', 'kvj', 'mmn', 'mmo', 'kgk', 'pbb', 'cbs', 'emp', 'zyp'}

def available_uriel_languages():
    avail = set()
    #for feature_set in FEATURE_SETS_DICT:
    for feature_set in ["fam"]:
        filename, source, prefix = FEATURE_SETS_DICT[feature_set]
        filename = pkg_resources.resource_filename(__name__, os.path.join('data', filename))
        feature_database = np.load(filename)
        mask = np.all(feature_database["data"] != -1.0, axis=0)
        langs = [feature_database["langs"][i] for i,m in enumerate(mask) if np.sum(m)>0]
        for l in langs:
            avail.add(l)
    return avail
    #return set([LETTER_CODES[lang_code] for lang_code in LETTER_CODES])

def available_learned_languages():
    return set(LEARNED_LETTER_CODES)

def available_languages():
    s1 = available_uriel_languages()
    s2 = available_learned_languages()
    return s1.union(s2)

def available_feature_sets():
    return [key for key in FEATURE_SETS_DICT]

LANGUAGES = available_languages()
URIEL_LANGUAGES = available_uriel_languages()
LEARNED_LANGUAGES = available_learned_languages()
FEATURE_SETS = available_feature_sets()

def get_language_code(lang_code, feature_database):
    # first, normalize to an ISO 639-3 code
    if lang_code in LETTER_CODES:
        lang_code = LETTER_CODES[lang_code]
    if lang_code not in feature_database["langs"]:
        message = "Note: Language " + lang_code + " not found in the URIEL database. "
        message += "Run lang2vec.LANGUAGES or lang2vec.available_languages()  to see a list of URIEL-supported languages. "
        if lang_code in LEARNED_LANGUAGES:
            message += "\nOnly a 'learned' feature vector is available.\n"
            message += "(run lang2vec.LEARNED_LANGUAGES or lang2vec.available_learned_languages() for a list of supported languages)"
            print(message)
            return "not_found"
        raise Exception(message)
    return lang_code

def get_learned_language_code(lang_code, feature_database):
    # first, normalize to an ISO 639-3 code
    if lang_code in LETTER_CODES:
        lang_code = LETTER_CODES[lang_code]
    if lang_code not in feature_database["langs"]:
        if lang_code in URIEL_LANGUAGES:
            print("Note: Language " + lang_code + " not found in the 'learned' feature set."+
                " However, it is available in the URIEL feature sets.")
            return "not_found"
        else:
            raise Exception("ERROR: Language " + lang_code + " not found in the learned or the URIEL feature database. " +
                "Run lang2vec.LANGUAGES or lang2vec.available_languages()" + 
                " to see a list of supported languages for the other feature sets." + 
                "Run lang2vec.LEARNED_LANGUAGES or lang2vec.available_learned_languages()" + 
                " to see a list of supported languages for the learned feature set.")
    return lang_code


def get_language_index(lang_code, feature_database):
    if lang_code == "not_found":
        return -1
    return np.where(feature_database["langs"] == lang_code)[0][0]
    
def get_source_index(source_name, feature_database):
    return np.where(feature_database["sources"] == source_name)[0]

def get_feature_names(feature_name_prefix, feature_database):
    return [ f for f in feature_database["feats"] if f.startswith(feature_name_prefix) ]

def get_feature_index(feature_name, feature_database):
    return np.where(feature_database["feats"] == feature_name)[0][0]
    
def get_id_set(lang_codes):
    #feature_database = np.load("family_features.npz")
    filename = "family_features.npz"
    filename = pkg_resources.resource_filename(__name__, os.path.join('data', filename))
    feature_database = np.load(filename)
    lang_codes = [ get_language_code(l, feature_database) for l in lang_codes ]
    all_languages = list(feature_database["langs"])
    feature_names = [ "ID_" + l.upper() for l in all_languages ]
    values = np.zeros((len(lang_codes), len(feature_names)))
    for i, lang_code in enumerate(lang_codes):
        feature_index = get_language_index(lang_code, feature_database)
        values[i, feature_index] = 1.0
    return feature_names, values

def get_learned_set(lang_codes):
    filename = "learned.npy"
    filename = pkg_resources.resource_filename(__name__, os.path.join('data', filename))
    feature_database = np.load(filename, encoding="latin1").item()
    lang_codes = [ get_learned_language_code(l, feature_database) for l in lang_codes ]
    feature_names = [ "LEARNED_%03d" % i for i in range(512) ]
    feature_values = np.ones((len(lang_codes),512))*(-1)
    for i, lang_code in enumerate(lang_codes):
        if lang_code != "not_found":
            feature_values[i] = feature_database[lang_code]

    return feature_names, feature_values

    
def get_named_set(lang_codes, feature_set):
    if feature_set == 'id':
        return get_id_set(lang_codes)
    if feature_set == "learned":
        return get_learned_set(lang_codes)
    
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
    feature_values = np.ones((len(lang_indices), len(feature_indices), len(source_index)))*(-1)
    feature_values = feature_database["data"][lang_indices,:,:][:,feature_indices,:][:,:,source_index]
    for i,l in enumerate(lang_indices):
        if l == -1:
            feature_values[i] = np.ones(features_values[i].shape)*(-1)
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