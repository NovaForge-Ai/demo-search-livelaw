
import os
import re
from flask import Flask, request, render_template_string, g
from elasticsearch import Elasticsearch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

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


def build_query(query_texts, query_texts_no_stop):
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
    for idx in range(len(query_texts)):
        functions.extend(
            [
        {
          "filter": {
            "match_phrase": {
              "document_text": {
                "query": query_texts[idx],
                "slop": 5
              }
            }
          },
          "weight": 1000
        },
        {
          "filter": {
            "match_phrase": {
              "document_text": {
                "query": query_texts_no_stop[idx],
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
                "query": query_texts_no_stop[idx],
              }
            }
          },
          "weight": 100
        },
        {
          "filter": {
            "match": {
              "document_text": {
                "query": query_texts_no_stop[idx],
                "operator": "AND",
                "fuzziness": "AUTO"
              }
            }
          },
          "weight": 2
        },
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
                        "query": query_text_no_stop,
                        "operator": "AND",
                        "fuzziness": "AUTO:4,7",
                      }
                    }
                  }
                  for query_text_no_stop in query_texts_no_stop
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
    "text": query_texts[-1],
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
        "fragment_size": 70,
        "number_of_fragments": 15,
        "type": "unified",
        "matched_fields": [
          "document_text"
        ],
        "highlight_query": {
            "bool": {
                "should": [
                {
                    "match_phrase": {
                    "document_text": {
                        "query": query_texts[-1],
                        "slop": 5,
                        "boost": 4
                    }
                    }
                },
                {
                    "match_phrase": {
                    "document_text": {
                        "query": query_texts_no_stop[-1],
                        "slop": 5,
                        "boost": 2
                    }
                    }
                },
                {
                    "match": {
                    "document_text": {
                        "query": query_texts_no_stop[-1],
                        "boost": 2
                    }
                    }
                },
                {
                    "match": {
                    "document_text": {
                        "query": query_texts_no_stop[-1],
                        "operator": "AND",
                        "fuzziness": "AUTO:4,7",
                        "boost": 1
                    }
                    }
                }
                ]
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
    if corrected_terms:
        corrected_query = " ".join(corrected_terms)
    if corrected_query.lower() == query.lower():
        corrected_query = ""
    return corrected_query

def get_snippets(highlights, corrected_query, query):
    all_snippets = []
    corrected_query_words = remove_stopwords(corrected_query).lower().split() if corrected_query else []
    query_words = remove_stopwords(query).lower().split()
    valid_words = set(query_words + corrected_query_words)

    for fragments in highlights.values():
        all_snippets.extend(fragments)

    updated_snippets = []
    for snippet in all_snippets:
        cleaned_snippet = snippet
        for match in re.finditer(r"<em>(.*?)</em>", snippet):
            highlighted_text = match.group(1)
            words = highlighted_text.lower().split()
            if not any(word in valid_words for word in words):
                cleaned_snippet = cleaned_snippet.replace(f"<em>{highlighted_text}</em>", highlighted_text)
        updated_snippets.append(cleaned_snippet)

    # Highlight <em> tags with styling
    formatted_snippets = [
        snippet.replace("<em>", "<mark style='background-color: #ffff00; font-weight: bold;'>").replace("</em>", "</mark>")
        for snippet in updated_snippets
    ]

    return formatted_snippets

def get_document_url(hit):
    url = hit["_source"].get("document_url", "")
    if len(url) == 0:
        return url
    return f"{JUDGMENT_PAGE_URL}{url}"

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    es = get_es()
    corrected_query = ""
    corrected_search = ""
    queries_text = request.values.get("query", "")
    if request.method == "POST":
        queries_text = request.form["query"]
    if queries_text:
        queries = [query.strip() for query in queries_text.split(',')]
        queries_no_stop = [remove_stopwords(query)for query in queries]
        es_query = build_query(queries, queries_no_stop)
        response = es.search(index="doc_zeta", body=es_query)
        corrected_query = get_corrected_query(response, queries[-1])
        for hit in response["hits"]["hits"]:
            highlights = hit.get("highlight", {})
            snippets = get_snippets(highlights, corrected_query, queries[-1])
            document_url = get_document_url(hit)

            results.append({
                "score": hit["_score"],
                "date": hit["_source"].get("document_date", "Date not available"),
                "case_name": hit["_source"].get("case_name", "Case name not available"),
                "document_url": document_url,
                "snippets": snippets
            })
        queries[-1] = corrected_query
        corrected_search = ", ".join(queries)
    return render_template_string(TEMPLATE, results=results, query=queries_text, corrected_query=corrected_query, corrected_search=corrected_search)

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
    <input type="text" name="query" placeholder="Enter search text" value="{{ query }}">
    <input type="submit" value="Search">
  </form>
    {% if corrected_query %}
    <p>Did you mean: 
        <a href="/?query={{ corrected_search | urlencode }}">
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
    app.run(debug=True)
