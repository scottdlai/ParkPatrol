# Welcome to ParkPatrol

### Setup

This project was developed and tested with `python 3.11`. There might be
deprecated warning for usage of `cgi`.

```sh
pip install -r requirements.txt

cd front-end
npm i
```

### Running the Application

This project has 2 parts:

- A web application that allows users to upload photos to interact with the
  model
- A backend server that has an API to interact with the model

Run the server

```
python server.py
```

You will have a web server run on port `8000`.

Start the web application in `front-end`

```
npm run dev
```

Your web application will be running on port `5173`.
