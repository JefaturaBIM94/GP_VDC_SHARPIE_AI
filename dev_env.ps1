$env:PYTHONPATH="$PWD\external\Depth-Anything-V2;$PWD"
uvicorn api_server:app --reload --port 8000
