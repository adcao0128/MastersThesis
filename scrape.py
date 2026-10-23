import requests
from bs4 import BeautifulSoup

url = "https://example.com/page-with-select"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find the select by name or id
select = soup.find("select", {"id": "my-select-id"})

# Extract all option values and text
options = select.find_all("option")
for option in options:
    value = option.get("value")
    text = option.text.strip()
    print(f"{value}: {text}")