
import base64
from functools import wraps
import json
import os
from typing import List, Tuple
from flask import Flask, request, render_template_string, g, Response
from elasticsearch import Elasticsearch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from dotenv import load_dotenv
load_dotenv()

from llm import ChatGPTTupleArrayFetcher, SearchHighlightResponse

app = Flask(__name__)

CLOUD_ID = os.environ.get('ELASTIC_CLOUD_ID')
USERNAME = os.environ.get('ELASTIC_USERNAME')
PASSWORD = os.environ.get('ELASTIC_PASSWORD')
JUDGMENT_PAGE_URL = "https://thejudgements.in/searchResult?url="

def check_auth(username, password):
    return (
        username == os.environ.get("APP_USERNAME") and
        password == os.environ.get("APP_PASSWORD")
    )

def authenticate():
    return Response(
        "Could not verify your access level.\n"
        "You have to login with proper credentials", 401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def get_es():
    if 'es' not in g:
        g.es = Elasticsearch(
            cloud_id=CLOUD_ID,
            basic_auth=(USERNAME, PASSWORD),
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
    return g.es

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word.lower() not in STOPWORDS)

class Term:
    text: str
    no_stop: str

    def __init__(self, text):
        self.text = text
        self.no_stop = remove_stopwords(text)
    
    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "no_stop": self.no_stop}

    @staticmethod
    def from_dict(data: dict[str, str]):
        term = Term(data["text"])
        term.no_stop = data["no_stop"]  # Skip recomputing
        return term

class SearchHighlightResponseWithNoStop:
    search: list[list[Term]]
    highlight: list[Term]

    def __init__(self, search: list[list[Term]], highlight: list[Term]):
        self.search = search
        self.highlight = highlight

    @staticmethod
    def from_search_highlight_response(resp: SearchHighlightResponse):
        search = [[Term(text) for text in group] for group in resp.search]
        highlight = [Term(text) for text in resp.highlight]
        return SearchHighlightResponseWithNoStop(search=search, highlight=highlight)


    def to_dict(self):
        return {
            "search": [[term.to_dict() for term in group] for group in self.search],
            "highlight": [term.to_dict() for term in self.highlight],
        }

    @staticmethod
    def from_dict(data: dict):
        search = [[Term.from_dict(term) for term in group] for group in data["search"]]
        highlight = [Term.from_dict(term) for term in data["highlight"]]
        return SearchHighlightResponseWithNoStop(search=search, highlight=highlight)

def encode_queries(queries: List[Tuple[str, SearchHighlightResponseWithNoStop]]) -> str:
    """Encodes a list of search highlight responses with no stop to a URL-safe string"""
    serializable = [(query, resp.to_dict()) for query, resp in queries]
    json_str = json.dumps({"resp": serializable}, separators=(",", ":"))
    return base64.urlsafe_b64encode(json_str.encode()).decode()

def decode_queries(encoded_str: str) -> List[Tuple[str, SearchHighlightResponseWithNoStop]]:
    """Decodes the URL-safe string back to a list of search highlight responses with no stop"""
    json_str = base64.urlsafe_b64decode(encoded_str.encode()).decode()
    data = json.loads(json_str)
    return [(query, SearchHighlightResponseWithNoStop.from_dict(resp)) for query, resp in data["resp"]]


def build_query(queries: list[list[list[Term]]], highlights: list[Term], plain_query: str):
    functions = [
        {
          "gauss": {
            "document_date": {
              "origin": "now",
              "scale": "300d",
              "decay": 0.99
            }
          }
        }
    ]
    
    highlight_functions = []
    for query in queries:
        for phrases in query:
            for phrase in phrases:
                highlight_functions.extend(
                    [
                        {
                    "match_phrase": {
                        "document_text": {
                        "query": phrase.text,
                        "slop": 20,
                        "boost": 10
                        }
                    }
                    },
                    {
                    "match_phrase": {
                        "document_text": {
                        "query": phrase.no_stop,
                        "slop": 20,
                        "boost": 5
                        }
                    }
                    }
                    ]
                )
    for highlight in highlights:
        if not highlight.no_stop:
            continue
        highlight_functions.extend(
            [
                {
                    "match_phrase": {
                        "document_text": {
                            "query": highlight.no_stop,
                            "slop": 20,
                            "boost": 1
                            }
                        }
                }
            ]
        )
    filter = []
    for query in queries:
        for phrases in query:
            filter.append(
                    {
                    "bool": {
                        "should": [
                        {
                            "match_phrase": {
                            "document_text": {
                                "query": phrase.no_stop,
                                "slop": 20
                            }
                            }
                        } for phrase in phrases
                        ],
                        "minimum_should_match": 1
                    }
                    }
            )
    return {
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "must": filter
        }
      },
      "functions": functions,
      "score_mode": "multiply",
      "boost_mode": "replace"
    }
  },
    "suggest": {
    "text": plain_query,
    "spellcheck": {
        "term": {
            "field": "document_text",
            "suggest_mode": "always"
        }
    }
    },
  "highlight": {
    "fields": {
      "document_text": {
        "fragment_size": 50,
        "number_of_fragments": 18,
        "type": "unified",
        "matched_fields": [
          "document_text"
        ],
        "highlight_query": {
            "bool": {
                "should": highlight_functions
            }
            }
      }
    },
    "fragmenter": "score_ordered",
    "require_field_match": True
  },
  "size": 50
}

def get_corrected_query(response, query):
    suggestions = response.get("suggest", {}).get("spellcheck", [])

    corrected_terms = []
    for entry in suggestions:
        options = entry.get("options", [])
        if options and ((options[0]["freq"] > 5 and options[0]["score"] > 0.9) or (options[0]["freq"] > 50 and options[0]["score"] > 0.7)):
            corrected_terms.append(options[0]["text"])
        else:
            corrected_terms.append(entry.get("text"))
    corrected_query = " ".join(corrected_terms)
    if corrected_query.lower() == query.lower():
        corrected_query = ""
    return corrected_query

def get_snippets(highlights):
    all_snippets = []

    for fragments in highlights.values():
        all_snippets.extend(fragments)

    # Highlight <em> tags with styling
    formatted_snippets = [
        snippet.replace("<em>", "<mark style='background-color: #ffff00; font-weight: bold;'>").replace("</em>", "</mark>")
        for snippet in all_snippets
    ]

    return formatted_snippets

def get_document_url(hit):
    url = hit["_source"].get("document_url", "")
    if len(url) == 0:
        return url
    return f"{JUDGMENT_PAGE_URL}{url}"

def get_llm():
    if 'llm' not in g:
        g.llm = ChatGPTTupleArrayFetcher()
    return g.llm

@app.route("/", methods=["GET", "POST"])
@requires_auth
def search():
    results = []
    es = get_es()
    llm = get_llm()
    corrected_query = ""
    previous_queries_encoded = request.values.get("previous_queries_encoded", "[]")
    query_text = request.values.get("new_query_text", "")
    display_query_text = ""
    if request.method == "POST":
        query_text = request.form["new_query_text"]
        previous_queries_encoded = request.form.get("queries", encode_queries([]))
    if query_text:
        queries = decode_queries(previous_queries_encoded)
        res = llm.fetch_search_highlight(query_text)
        if res is None:
            res = SearchHighlightResponse([[query_text]], [query_text])
        res_with_no_stop = SearchHighlightResponseWithNoStop.from_search_highlight_response(res)
        queries.append((query_text, res_with_no_stop))
        es_query = build_query([query.search for _, query in queries], res_with_no_stop.highlight, query_text)
        response = es.search(index="doc_zeta", body=es_query)
        corrected_query = get_corrected_query(response, query_text)
        for hit in response["hits"]["hits"]:
            highlights_res = hit.get("highlight", {})
            snippets = get_snippets(highlights_res)
            document_url = get_document_url(hit)
            results.append({
                "score": hit["_score"],
                "date": hit["_source"].get("document_date", "Date not available"),
                "case_name": hit["_source"].get("case_name", "Case name not available"),
                "document_url": document_url,
                "snippets": snippets
            })
        previous_queries_encoded = encode_queries(queries)
        display_query_text = ", ".join([text for text, _ in queries])
    return render_template_string(TEMPLATE, results=results, new_query_text="", corrected_query=corrected_query, previous_queries_encoded=previous_queries_encoded, display_query_text=display_query_text)

TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Search</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    input[type="text"] { width: 400px; padding: 8px; }
    input[type="submit"] { padding: 8px 16px; }
    .result { margin-bottom: 20px; padding: 12px; border: 1px solid #ccc; border-radius: 6px; }
    .highlight { background-color: #ffff00; font-weight: bold; }
  </style>
</head>
<body>
  <h2>Search Elasticsearch</h2>
  <form method="post">
    <input type="text" name="new_query_text" placeholder="Enter search text" value="{{ new_query_text }}">
    <input type="hidden" name="previous_query" value="{{ previous_queries_encoded }}">
    <input type="submit" value="Search">
  </form>
    {% if display_query_text %}
    <p>
    Current Searches: {{display_query_text}}
    <p>
    {% endif %}
    {% if not display_query_text %}    <p> <p>
    {% endif %}
  <form method="get" action="/">
    <button type="submit">Reset</button>
  </form>
    {% if corrected_query %}
    <p>Did you mean: 
        <a>
        <strong>{{ corrected_query }}</strong>
        </a>?
    </p>
    {% endif %}
  {% if results %}
    <h3>Results:</h3>
    <ul>
    {% for r in results %}
      <li class="result">
        <div><strong>Case Name:</strong> 
            {% if r.document_url %}
                <a href="{{ r.document_url }}" target="_blank">{{ r.case_name }}</a>
            {% else %}
                {{ r.case_name }}
            {% endif %}
        </div>
        <div><strong>Score:</strong> {{ r.score }}</div>
        <div><strong>Date:</strong> {{ r.date }}</div>
        {% for snippet in r.snippets %}
          <div>{{ snippet|safe }}</div>
        {% endfor %}
      </li>
    {% endfor %}
    </ul>
  {% endif %}
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get("PORT") or 5001, debug=True)