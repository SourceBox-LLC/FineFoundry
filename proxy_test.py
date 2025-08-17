import requests, random, time

PROXY = "socks5h://127.0.0.1:9050"
PROXIES = {"http": PROXY, "https": PROXY}

UAS = [
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
]

s = requests.Session()
s.proxies.update(PROXIES)

def get(url, **kw):
    headers = {"User-Agent": random.choice(UAS)}
    return s.get(url, headers=headers, timeout=25, **kw)

# sanity check
print(get("https://httpbin.org/ip").json())
