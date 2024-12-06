import google.auth.transport.requests
import google.oauth2.id_token
import requests

def invoke_training_service(url):
    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, url)
    headers = {'Authorization': f'Bearer {id_token}'}
    response = requests.post(url, headers=headers)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    SERVICE_URL = 'https://cloud-run-training-349886376829.us-central1.run.app' 
    invoke_training_service(SERVICE_URL)
