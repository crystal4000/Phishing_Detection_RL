# url_parser.py
# %%writefile url_parser.py
import tldextract
import urllib.parse
import re

def parse_url(url):
    # Use tldextract to parse the domain
    extracted = tldextract.extract(url)

    # Use urllib to parse the full URL
    parsed = urllib.parse.urlparse(url)

    # Extract query parameters
    query_params = urllib.parse.parse_qs(parsed.query)

    # Identify the protocol
    protocol = parsed.scheme if parsed.scheme else 'http'

    # Extract path components
    path_components = [comp for comp in parsed.path.split('/') if comp]

    # Identify if it's an IP address
    is_ip = bool(re.match(r'\d+\.\d+\.\d+\.\d+', extracted.domain))

    # Create a dictionary with all the parsed information
    parsed_data = {
        'url': url,
        'protocol': protocol,
        'subdomain': extracted.subdomain,
        'domain': extracted.domain,
        'tld': extracted.suffix,
        'registered_domain': extracted.registered_domain,
        'path': parsed.path,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'port': parsed.port,
        'is_ip': is_ip,
        'path_components': path_components,
        'query_params': query_params
    }

    return parsed_data

# # Example usage
# if __name__ == "__main__":
#     sample_url = "https://sub.example.com:8080/path/to/page.html?param1=value1&param2=value2#section"
#     parsed_result = parse_url(sample_url)

#     print("Parsed URL Data:")
#     for key, value in parsed_result.items():
#         print(f"{key}: {value}")