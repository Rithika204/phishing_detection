<<<<<<< HEAD
# utils.py
import re
import numpy as np
from urllib.parse import urlparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def extract_basic_features(url):
    """Extract numeric + keyword features for XGBoost"""
    features = {}
    features['url_length'] = len(url)
    features['has_https'] = int(url.startswith('https'))
    features['has_ip'] = int(bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", url)))
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special'] = len(re.findall(r'[^\w]', url))
    features['num_subdomains'] = url.count('.')
    features['has_at'] = int('@' in url)
    features['has_hyphen'] = int('-' in url)
    keywords = ['login','verify','secure','bank','update','account']
    features['suspicious_keywords'] = int(any(k in url.lower() for k in keywords))
    return np.array(list(features.values())).reshape(1, -1)


# -------- CNN text processing --------
def preprocess_for_cnn(url, tokenizer, maxlen=100):
    seq = tokenizer.texts_to_sequences([url])
    return pad_sequences(seq, maxlen=maxlen)
=======
# utils.py
import re
import numpy as np
from urllib.parse import urlparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def extract_basic_features(url):
    """Extract numeric + keyword features for XGBoost"""
    features = {}
    features['url_length'] = len(url)
    features['has_https'] = int(url.startswith('https'))
    features['has_ip'] = int(bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", url)))
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special'] = len(re.findall(r'[^\w]', url))
    features['num_subdomains'] = url.count('.')
    features['has_at'] = int('@' in url)
    features['has_hyphen'] = int('-' in url)
    keywords = ['login','verify','secure','bank','update','account']
    features['suspicious_keywords'] = int(any(k in url.lower() for k in keywords))
    return np.array(list(features.values())).reshape(1, -1)


# -------- CNN text processing --------
def preprocess_for_cnn(url, tokenizer, maxlen=100):
    seq = tokenizer.texts_to_sequences([url])
    return pad_sequences(seq, maxlen=maxlen)
>>>>>>> c9dc7fe (Initial commit)
