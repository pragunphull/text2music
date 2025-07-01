import json

with open('maestro-v3.0.0.json', 'r') as file:
    data = json.load(file)

print("Keys in the JSON dictionary:", data.keys())

# Print the entire dictionary (for inspection)
print("JSON content:", json.dumps(data, indent=4))

try:
    with open("maestro-v3.0.0.json", "r") as f:
        metadata_pairs = json.load(f)
    if isinstance(metadata_pairs, list):
        print(metadata_pairs[:2])
    else:
        print("JSON is not a list. Structure:", type(metadata_pairs))
except Exception as e:
    print("Error loading JSON file:", e)

