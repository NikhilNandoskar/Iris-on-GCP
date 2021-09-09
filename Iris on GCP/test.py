# Imports
import requests

# Input Dictionary
payload = {'sepal_length': 4.6, 'sepal_width': 4.2,
        'petal_length': 1, 'petal_width': 2.2,
        'model_name':'Logistic Regression'}

#response = requests.get('http://127.0.0.1:5000/test', params = payload)

response = requests.get('https://predictiristype-4il23ksazq-uk.a.run.app/test', params=payload)

print('Status code: {}'.format(response.status_code))
print('Payload:\n{}'.format(response.text))

