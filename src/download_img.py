import requests
import os
import env
class VerifyEmail(Exception):
    pass
class NoEmailError(Exception):
    pass
class RegistrationError(Exception):
    pass


def register(email, name, description):
    REGISTER_URL = 'https://api.openverse.org/v1/auth_tokens/register/'

    data = {'name': name,
            'description': description,
            'email': email}

    response = requests.post(REGISTER_URL, json=data)

    if response.status_code == 201:

        client_data = response.json()
        client_id = client_data['client_id']
        client_secret = client_data['client_secret']

        save_to_env(email, name, description, client_id, client_secret)
        raise VerifyEmail("check email to verify for better performance")
    else:
        print('Registration failed:', response.text)
        return None, None

def save_to_env(email, name, description, client_id, client_secret):
    with open('.env', 'w') as f:
        f.write(f'EMAIL={email}\n' if email is not None else 'EMAIL=\n')
        f.write(f'NAME={name}\n' if name is not None else 'NAME=\n')
        f.write(f'DESCRIPTION={description}\n' if description is not None else 'DESCRIPTION=\n')
        f.write(f'CLIENT_ID={client_id}\n' if client_id is not None else 'CLIENT_ID=\n')
        f.write(f'CLIENT_SECRET={client_secret}\n' if client_secret is not None else 'CLIENT_SECRET=\n')

def load_from_env():
    email = 'u6774804@anu.edu.au'
    name = 'NewsAggregator'
    description = 'This is a sample project description'
    client_id = 'SQXoubtYPuNZqo9cA4d85LYmHollCwI02lgysNu3'
    client_secret = 'ukV6KaKiQmD2sHpyGjRdbXrJKDV5SCfvWHY1hysZQG5ygSRwdzTo5GsaZZ7ihnxILgslo06iDeg3ZArme01JSJYrpmalb0VDsijVPtdYFQZmXCbfVe1CJ0jkiP1ocFez'
    return email, name, description, client_id, client_secret

email, name, description, client_id, client_secret = load_from_env()

if not (email and name and description and client_id and client_secret):
    print("Registering user...")
    email = os.getenv('EMAIL')
    if (email==None or name==None or description==None):
        save_to_env(email, name, description, client_id, client_secret)
        raise NoEmailError("no email or name or description in .env")
    client_id, client_secret = register(email, name, description)
    raise RegistrationError("no client id or secret")


def token(client_id, client_secret, grant_type="client_credentials"):
    TOKEN_URL = 'https://api.openverse.org/v1/auth_tokens/token/'

    data = {'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': grant_type}


    response = requests.post(TOKEN_URL, data=data)

    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Token request failed with status code {response.status_code}: {response.text}")

'''
print("Email:", email)
print("Client ID:", client_id)
print("Client Secret:", client_secret)
'''
access_token = token(client_id, client_secret)

# print("access token:", access_token)

def key_info(access_token):
    KEY_INFO_URL = 'https://api.openverse.org/v1/rate_limit/'

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(KEY_INFO_URL, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Key info request failed with status code {response.status_code}: {response.text}")

rate_limit= key_info(access_token)

# print(rate_limit)


def images_search(access_token, term):
    IMAGES_SEARCH_URL = 'https://api.openverse.org/v1/images/'

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    params = {
        'page': 1,
        'page_size': 30,
        'q': term
    }

    response = requests.get(IMAGES_SEARCH_URL, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Image search request failed with status code {response.status_code}: {response.text}")


def download_image(image_url, folder_path):
    # Extract image file name from URL
    image_name = image_url.split('/')[-1]
    # Download the image and save it to the specified folder
    headers = {'User-Agent': "NewsAggregatorImageAgent/1.0"}

    response = requests.get(image_url, headers=headers)
    if response.status_code == 200:
        with open(os.path.join(folder_path, image_name), 'wb') as f:
            f.write(response.content)
            print(f"Image downloaded successfully: {image_name}")
    else:
        print(response)
        print(response.content)
        print(f"Failed to download image: {image_url}")


def return_image(image_url):
    # Extract image file name from URL
    image_name = image_url.split('/')[-1]
    # Download the image and save it to the specified folder
    headers = {'User-Agent': "NewsAggregatorImageAgent/1.0"}

    response = requests.get(image_url, headers=headers)
    if response.status_code == 200:
        print(f"Image downloaded successfully: {image_name}")
        return response.content
    else:
        print(response)
        print(response.content)
        print(f"Failed to download image: {image_url}")
        return None


def download_entity_image(entity):
    images=images_search(access_token, entity)
    entity_name= entity.replace(" ", "_")

    #create folder
    if not os.path.exists(os.path.join(env.faces_folder_base, entity_name)):
        os.makedirs(os.path.join(env.faces_folder_base, entity_name))

    for image in images['results']:
        title = image['title']
        download_image(image['url'], os.path.join(env.faces_folder_base, entity_name))

# download_entity_image("Justin Trudeau")
# print("download finished")
