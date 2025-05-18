
import os
from flask import Flask, request, render_template_string, g
from elasticsearch import Elasticsearch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from dotenv import load_dotenv
load_dotenv()

from llm import ChatGPTTupleArrayFetcher

app = Flask(__name__)

CLOUD_ID = os.environ.get('ELASTIC_CLOUD_ID')
USERNAME = os.environ.get('ELASTIC_USERNAME')
PASSWORD = os.environ.get('ELASTIC_PASSWORD')
JUDGMENT_PAGE_URL = "https://thejudgements.in/searchResult?url="


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

class Query:
    text: str
    no_stop: str
    priority: int

    def __init__(self, text, no_stop, priority):
        self.text = text
        self.no_stop = no_stop
        self.priority = priority


def build_query(queries: list[Query], highlights: list[Query]):
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
    for query in queries:
        if not query.text:
            continue
        functions.append({
          "filter": {
            "match_phrase": {
              "document_text": {
                "query": query.text,
                "slop": 5
              }
            }
          },
          "weight": 1000
        })
    for query in queries:
        if not query.no_stop:
            continue
        functions.extend(
            [
        {
          "filter": {
            "match_phrase": {
              "document_text": {
                "query": query.no_stop,
                "slop": 5
              }
            }
          },
          "weight": 500
        },
        {
          "filter": {
            "match": {
              "document_text": {
                "query": query.no_stop,
              }
            }
          },
          "weight": 10
        },
      ]
        )
    
    highlight_functions = []
    for highlight in highlights:
        if not highlight.no_stop:
            continue
        highlight_functions.extend(
            [
                {
                    "match_phrase": {
                        "document_text": {
                            "query": highlight.no_stop,
                            "slop": 5,
                            "boost": 5-highlight.priority
                            }
                        }
                }
            ]
        )

    return {
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "should": [
            {
              "bool": {
                "filter": [
                  {
                    "match": {
                      "document_text": {
                        "query": query.no_stop,
                        "operator": "AND",
                      }
                    }
                  }
                  for query in queries if query.no_stop
                ]
              }
            } 
          ],
          "minimum_should_match": 1
        }
      },
      "functions": functions,
      "score_mode": "multiply",
      "boost_mode": "replace"
    }
  },
    "suggest": {
    "text": queries[-1].text,
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

def get_queries(query_text: str, generated_previous_query: str, llm: ChatGPTTupleArrayFetcher) -> tuple[list[Query], list[Query]]:
    queries = []
    highlights = []
    generation_prompt = """
We have a search system that takes a query from a lawyer. They are looking for relevant cases. But we are doing keyword search. Our search accepts multiple phrases which are searched based on match_phrase with slop. 

We want to split it into phrases that appear in the judgements and have legal meaning. 

Include relevant sections of case laws as well. Can you give the list of these phrases in an array with its priority

Priority 1 should only contain terms inside the query

Priority 2 and 3 can have the different legal terms associated including acts and sections
Priority 2 and 3 can have words which are verbs or plurals like for terminating it can be terminated and terminates

for example: "denial of sanction after taking cognizance"

answer will be something like:

 [
  ("denial of sanction", 1),
  ("denials of sanctions", 2),
  ("taking cognizance", 1),
  ("sanction for prosecution", 2),
  ("cognizance without sanction", 2)
  ("cognizance of offence", 2),
  ("refusal to grant sanction", 2),
  ("absence of sanction", 2),
  ("invalid sanction", 3),
  ("requirement of sanction", 3),
  ("sanction under Section CrPC", 2),
  ("sanction under  Act", 2),
  ("bar  taking cognizance", 3)
]

Here we are removing stop words and we are adding slop, so you don't need to repeat words with different phrasing unless the word is not a non stop word or an entirely different way of phrasing it.

Only reply just the fixed list of tuple[str, int] in the format like [('text_1', 1), ('text_2', 2)]
"""
    fix_prompt = """Only reply just the fixed list of tuple[str, int] in the format like [('text_1', 1), ('text_2', 2)]"""
    pred_queries = llm.fetch_array_of_tuples(generation_prompt, query_text, fix_prompt)
    print(pred_queries)
    for pred_query, priority in pred_queries:
        no_stop = remove_stopwords(pred_query)
        if priority == 1:
            queries.append(Query(pred_query, no_stop, priority))
        highlights.append(Query(pred_query, no_stop, priority))
    for prev_query in [s.strip() for s in generated_previous_query.split(",")]:
        queries.append(Query(prev_query, remove_stopwords(prev_query), 1))
    return queries, highlights

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    es = get_es()
    llm = get_llm()
    corrected_query = ""
    previous_query = request.values.get("previous_query", "")
    generated_query = request.values.get("generated_query", "")
    query_text = request.values.get("query_text", "")
    if request.method == "POST":
        query_text = request.form["query_text"]
        previous_query = request.form.get("previous_query", "")
        generated_query = request.form.get("generated_query", "")
    if query_text:
        print(generated_query, query_text)
        queries, highlights = get_queries(query_text, generated_query, llm)
        es_query = build_query(queries, highlights)
        response = es.search(index="doc_zeta", body=es_query)
        corrected_query = get_corrected_query(response, queries[-1].text)
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
        previous_query = previous_query + ", " +  query_text
        previous_query = previous_query.strip(", ")
        generated_query = ", ".join([query.text for query in queries if query.text]) 
    return render_template_string(TEMPLATE, results=results, query_text="", corrected_query=corrected_query, previous_query=previous_query, genearted_query=generated_query)

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
    <input type="text" name="query_text" placeholder="Enter search text" value="{{ query_text }}">
    <input type="hidden" name="previous_query" value="{{ previous_query }}">
    <input type="hidden" name="generated_query" value="{{ generated_query }}">
    <input type="submit" value="Search">
  </form>
    {% if previous_query %}
    <p>
    Current Searches: {{previous_query}}
    <p>
    {% endif %}
    {% if not previous_query %}    <p> <p>
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