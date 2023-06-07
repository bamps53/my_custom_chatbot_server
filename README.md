# Custom Chatbot Server

Blog (Japanese only): https://note.com/mahlab/n/n99f28f55287b

## What is this for?

This is a FastAPI server for mocking the connection destination of
the [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui). By setting this server as the
connection destination, you can test chat responses from models implemented by yourself on a screen like ChatGPT. It is
also ideal for demonstrations.

## Setup

```
$ poetry install
$ poetry run python app/main.py
```


## Test

```
$ curl -X POST "http://127.0.0.1:11000/v1/chat/completions" -H "accept: text/event-stream" -H "Content-Type: application/json" --data '{"model": "simple-conversation-chat", "messages": [{"role": "system", "content": "You are a poetic assistant"}, {"role": "user", "content": "Write a poem about the ocean"}], "max_tokens": 1000, "temperature": 1, "stream": true}'
```