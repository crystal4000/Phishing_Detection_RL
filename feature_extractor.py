# feature_extractor.py
# %%writefile feature_extractor.py
import re
import urllib.parse
import tldextract

def extract_features(url):
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    extracted = tldextract.extract(url)

    # Initialize feature vector
    features = [0] * 14

    # 1. Length of URL
    features[0] = len(url)

    # 2. Length of domain
    features[1] = len(domain)

    # 3. Number of dots in domain
    features[2] = domain.count('.')

    # 4. Number of hyphens in domain
    features[3] = domain.count('-')

    # 5. Number of underscores in domain
    features[4] = domain.count('_')

    # 6. Number of forward slashes in path
    features[5] = path.count('/')

    # 7. Number of digits in domain
    features[6] = sum(c.isdigit() for c in domain)

    # 8. Presence of IP address in domain
    features[7] = int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain)))

    # 9. Presence of '@' symbol in URL
    features[8] = int('@' in url)

    # 10. Presence of double slash in path
    features[9] = int('//' in path)

    # 11. URL shortening service
    short_services = ['bit.ly', 'goo.gl', 't.co', 'tinyurl.com']
    features[10] = int(any(service in domain for service in short_services))

    # 12. Length of top-level domain
    features[11] = len(extracted.suffix)

    # 13. Presence of 'https' in protocol
    features[12] = int(parsed_url.scheme == 'https')

    # 14. Presence of common phishing words in domain
    phishing_words = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'signin', 'banking', 'confirm']
    features[13] = int(any(word in domain for word in phishing_words))

    return features

# # Example usage
# if __name__ == "__main__":
#     sample_url = "http://www.example.com/path/to/page.html"
#     features = extract_features(sample_url)
#     print(f"Features for {sample_url}:")
#     print(features)
