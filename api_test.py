import requests

data = {
    "list1": 1,
    "list2": 2,
    "list3": 3,
}
res = requests.get('http://127.0.0.1:5000/api/height', params=data)
print(res.json())

res = requests.get('http://127.0.0.1:5000/api/gender', params={
    "list1": 10,
    "list2": 10,
    "list3": 10,
})
print(res.json())



res = requests.get('http://127.0.0.1:5000/api/foot', params={
    "list1": 178,
    "list2": 80,
    "list3": 1,
})
print(res.json())

res = requests.get('http://127.0.0.1:5000/api/wine', params={
    'list1': 1,
    'list2': 1,
    'list3': 1,
    'list4': 1,
    'list5': 1,
    'list6': 1,
    'list7': 1,
    'list8': 1,
    'list9': 1,
    'list10': 1,
    'list11': 1,
    'list12': 1,
    'list13': 1,
})
print(res.json())
