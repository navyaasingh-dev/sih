import torch
import math
import tldextract
import idna

VALID_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
char_to_int = {char: idx + 1 for idx, char in enumerate(VALID_CHARS)}

def compute_lexical_features(domain_string: str):
    """Calculates Shannon entropy and length to catch dictionary-based DGAs."""
    try:
        domain_string = idna.decode(domain_string)
    except idna.IDNAError:
        pass
        
    extracted = tldextract.extract(domain_string)
    payload = f"{extracted.subdomain}.{extracted.domain}" if extracted.subdomain else extracted.domain
    payload = payload.lower()
    
    if not payload:
        return 0.0, 0.0
        
    entropy = 0.0
    for x in set(payload):
        p_x = float(payload.count(x)) / len(payload)
        entropy += - p_x * math.log(p_x, 2)
        
    return float(entropy), float(len(payload))

def domain_to_tensor(domain, max_len=253):
    try:
        domain = idna.decode(domain)
    except idna.IDNAError:
        pass
        
    extracted = tldextract.extract(domain)
    payload = f"{extracted.subdomain}.{extracted.domain}" if extracted.subdomain else extracted.domain
    payload = payload.lower()
    
    encoded = [char_to_int.get(char, 0) for char in payload[:max_len] if char in char_to_int]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
        
    return torch.tensor(encoded, dtype=torch.long)