# codif-ape-graph-rag

You need to install to first install [`uv`](https://github.com/astral-sh/uv), using for instance `pip install uv` if applicable.

Then run the following commands:

```bash
cd src
. ./setup.sh
export NEO4J_API_KEY= [NEO4J PASSWORD]
uvicorn api.main:app --reload --host 0.0.0.0 --port 5000
```
