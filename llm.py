from logging import getLogger
import openai
import ast
from typing import List, Tuple, Union
import os

class SearchHighlightResponse:
    search: list[list[str]]
    highlight: list[str]

    def __init__(self, search: list[list[str]], highlight: list[str]):
        self.search = search
        self.highlight = highlight
    
    @staticmethod
    def from_dict(data: dict):
        return SearchHighlightResponse(search=data["search"], highlight=data["highlight"])

class ChatGPTTupleArrayFetcher:
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        max_retries: int = 3
    ):
        self.model = model
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.max_retries = max_retries

    def fetch_array_of_tuples(
        self,
        user_query: str,
    ) -> Union[List[Tuple[str, int]], None]:
        """Main method to fetch an array of tuples from ChatGPT."""

        system_prompt = """
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

        l = getLogger()
        for attempt in range(self.max_retries):
            response_text = self._query_chatgpt(system_prompt, f"provide result for: '{user_query}'")
            parsed_array = self._try_parse_tuple_array(response_text)
            if parsed_array is not None:
                return parsed_array
            else:
                l.info(f"[Attempt {attempt+1}] Invalid tuple array format. Retrying with fix prompt.")
                user_query = f"Fix this output:\n{response_text}"
                system_prompt = fix_prompt
        l.info("Max retries reached. Failed to get valid tuple array.")
        return None

    def _query_chatgpt(self, system_prompt: str, user_query: str) -> str:
        """Query ChatGPT with system + user messages."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2
        )
        # Extract response content
        content = response.choices[0].message.content.strip()

        # Token usage info (if available)
        usage = getattr(response, "usage", None)
        prompt_tokens = usage.prompt_tokens if usage else "?"
        completion_tokens = usage.completion_tokens if usage else "?"
        total_tokens = usage.total_tokens if usage else "?"

        # Debug output
        l = getLogger()
        l.info(f"user query: {user_query}")
        l.info(f"input token consumption: {prompt_tokens}")
        l.info(f"output token consumption: {completion_tokens}")
        l.info(f"total token consumption: {total_tokens}")
        l.info("-------LLM Response Start-------")
        l.info(content)
        l.info("-------LLM Response End-------")

        return content

    def _try_parse_tuple_array(self, output: str) -> Union[List[Tuple], None]:
        """Try to parse the output as a list of tuples using ast.literal_eval."""
        l = getLogger()
        try:
            output = output.replace("\n", "")
            parsed = ast.literal_eval(output)
            if isinstance(parsed, list) and all(isinstance(item, tuple) and isinstance(item[0], str) and isinstance(item[1], int) for item in parsed):
                return parsed
            else:
                l.info("Parsed result is not a list of tuples.")
                return None
        except Exception as e:
            l.info(f"Failed to parse output as Python list of tuples: {e}")
            return None
        
    def _try_parse_search_highlight_json(self, output: str) -> SearchHighlightResponse | None:
        """Try to parse json as SearchHighlightResponse"""
        l = getLogger()
        try:
            parsed = ast.literal_eval(output)
            if (
                isinstance(parsed, dict)
                and "search" in parsed
                and "highlight" in parsed
                and isinstance(parsed["highlight"], list)
                and all(isinstance(item, str) for item in parsed["highlight"])
                and isinstance(parsed["search"], list)
                and all(isinstance(row, list) and all(isinstance(col, str) for col in row) for row in parsed["search"])
            ):
                return SearchHighlightResponse(parsed["search"], parsed["highlight"])
            else:
                l.info("Parsed result is not a SearchHighlightResponse.")
                return None
        except Exception as e:
            l.info(f"Failed to parse output as SearchHighlightResponse: {e}")
            return None
        
    def fetch_search_highlight(
        self,
        user_query: str,
        section: dict,
    ) -> Union[SearchHighlightResponse, None]:
        """Main method to fetch SearchHighlight from ChatGPT."""
        if section:
            user_query += f" ; {section['title']}"
            print(user_query)
#         system_prompt = """
# You are helping improve a legal document search system that uses keyword-based phrase search with slop. Your objective is to output phrases that are present in judgements to better match the contents of what the user queried for.

# Given a user's legal query, you need to:
# 1. Extract important phrases that have legal meaning. Rephrase them to include how they might be used in a judgement. expand abbreviations when necesasry.
# 2.If the query has some text after ";" sign, it is the explanation of the act or section or article mentioned in the userquery. You may use this information as well for rephrasing
# 3. Group those phrases into a 2D array (list of list of strings). Each sublist represents a group of semantically related phrases that can be searched together. All groupings should have atleast one term directly from the search, you can correct spelling mistakes though only if its very evident. Add plurals and different wording within each group. If the query mentions an act,you can find the section title after the ';' in the query. rephrase that in ways that it might appear in the judgement.
# 4. Additionally, highlight the most critical phrases (from the user query or closely related) in a separate list. Legal terms associated including acts and sections. You can use the phrases from the explanation given as well.

# Return a **JSON-like Python dictionary** with:
# - "search": a list of lists of phrases (each phrase is a string).
# - "highlight": a list of key phrases to visually highlight to the user.

# All entries must be strings. Avoid empty lists.

# Example input query: "denial of sanction after taking cognizance"

# Example output:
# {
#   "search": [
#     ["denial of sanction", "refusal to grant sanction", "denial of sanctions"],
#     ["taking cognizance", "cognizance of offence", "cognizance without sanction", "bar to taking cognizance"]
#   ],
#   "highlight": [
#     "sanction under Section",
#     "absence of sanction",
#     "invalid sanction",
#     "requirement of sanction"
#   ]
# }

# Example input query: "65 of IT Act ; Tampering with computer source documents"

# Example output:
# {
#     "search": [
#         ["Section 65 of IT Act", "Section 65 of Information Technology Act","Section 65 of Income Tax Act", "tampering with computer source documents"],
#         ["IT Act", "Income Tax Act","Information Technology Act"]
#     ],
#     "highlight": [
#         "Section 65 IT Act",
#         "tampering with computer source documents",
#         "computer source code"
#     ]
# }

# Example input query: "heydons law"

# Example output:
# {
#     "search": [
#         ["Heydon's case", "Heydon's Rule", "mischief rule"]
#     ],
#     "highlight": [
#         "purposive interpretation",
#         "statutory interpretation"
#     ]
# }

# Example input query: "murder"

# Example output:
# {
#     "search": [
#         ["murder", "murders", "homicide", "homicides", "manslaughter", "manslaughters","killed"]
#     ],
#     "highlight": [
#         "first degree murder",
#         "second degree murder",
#         "voluntary manslaughter",
#         "involuntary manslaughter"
#     ]
# }
# Example input query: "supreme court on adoption laws"

# Example output:
# {
#     "search": [
#         ["adoption laws", "adoption statutes", "adoption regulations","child adoption laws", "laws on adopting a child"],
#         ["child adoption"],
#         ["supreme court", "sc", ]
#     ],
#     "highlight": [
#         "legal requirements for adoption",
#         "adoption eligibility",
#         "adoption process"
#     ]
# }

# Example input query: "defence witness to be treated on par with prosecution witness"

# Example output:
# {
#     "search": [
#         ["defence witness treated on par prosecution witness", "defence witness equitable treatment prosecution witness", "defence witness equal treatment prosecution witness","treatment of witness"],
#     ],
#     "highlight": [
#         "witness credibility",
#         "witness impartiality",
#         "rights of defense witness",
#         "rights of prosecution witness"
#     ]
# }
# Example input query: "victim was decapitated"

# Example output:
# "search": [
#         ["victim was decapitated", "victim decapitation", "decapitated victim","neck was cut off","head cut"]
#     ],
#     "highlight": [
#         "homicide by decapitation",
#         "murder by decapitation",
#         "aggravated murder",
#         "felony murder"
#     ]
# }

# Only return the Python dictionary in this format. Do not add explanations or extra text.
# """

#         """Example input query: "section 17 of the registration act"

# Example output:
# {
#     "search": [
#         [ "Section 71 Registration Act", "s. 71 Registration Act","Reasons for refusal to register","recorded refusal to register"]
#     ],
#     "highlight": [
#         "rectification of errors under Section 71",
#         "powers under Section 71",
#         "Registration Act Section 71"
#     ]
# }
#         """
        system_prompt = """
You are assisting in enhancing a legal judgment search engine. Your task is to transform user-submitted legal queries into effective search phrases, ensuring these phrases closely match the wording commonly found in Supreme Court judgments.

**Task Instructions:**

1. **Extract Key Legal Phrases:**
   Identify significant legal phrases from the user's query. Rephrase them explicitly as they might appear in Supreme Court judgments. Expand abbreviations and correct obvious spelling mistakes only if necessary.

2. **Utilize Explanatory Information (after ';'):**
   If the user's query contains explanatory text after a ";", leverage this information to clarify and rephrase phrases, particularly sections or acts, into wording typically used within judgments.

3. **Semantic Grouping:**
   Organize these phrases into semantically related groups. Each group (sublist) should include at least one phrase directly from the user's original query. Within each group, provide plural forms and closely related alternative phrasings.

4. **Highlight Critical Phrases:**
   Separately list key legal terms and phrases (especially acts, sections, or clearly emphasized legal points) for visual emphasis. Prioritize phrases directly referenced or closely related to the user's query.

**Output Requirements:**
Return only a valid Python dictionary formatted exactly as follows:

```python
{
  "search": [["phrase1", "alternative phrase1"], ["phrase2", "alternative phrase2"], ...],
  "highlight": ["key phrase1", "key phrase2", ...]
}
```

**Ensure:**

* All elements are strings.
* No empty lists.
* Exactly two keys: `"search"` and `"highlight"`.
* No extra comments, explanations, or syntax errors.

**Examples:**

Example Input query: "denial of sanction after taking cognizance"

Example output:
{
  "search": [
    ["denial of sanction", "refusal to grant sanction", "denial of sanctions"],
    ["taking cognizance", "cognizance of offence", "bar to taking cognizance"]
  ],
  "highlight": [
    "sanction under Section",
    "absence of sanction",
    "invalid sanction",
    "requirement of sanction"
  ]
}
```

Example Input query: "65 of IT Act ; Tampering with computer source documents"

Example output:
{
  "search": [
    ["Section 65 of IT Act", "Section 65 Information Technology Act", "tampering with computer source documents", "computer source code tampering"],
    ["IT Act", "Information Technology Act"]
  ],
  "highlight": [
    "Section 65 IT Act",
    "tampering with computer source documents",
    "computer source code"
  ]
}
```
Example Input query: "section 18 of registration act ; documents of which registration is optional."

Example output:
{
  "search": [
    ["section 18 of registration act", "documents of which registration is optional","optionally registrable"],
    ["registration Act"]
  ],
  "highlight": [
    "Section 17 Registration Act",
    "optional registration",
    "optionally registrable"
  ]
}
```

example Input query: "heydons law"

Example output:
{
  "search": [
    ["Heydon's case", "Heydon's Rule", "mischief rule"]
  ],
  "highlight": [
    "purposive interpretation",
    "statutory interpretation"
  ]
}
```

Example input query: "victim was decapitated"
Example output:
{
  "search": [
    ["victim was decapitated", "victim decapitation", "head cut off", "neck was severed"]
  ],
  "highlight": [
    "homicide by decapitation",
    "murder by decapitation",
    "aggravated murder",
    "felony murder"
  ]
}
```

Follow this format strictly and ensure phrases match judicial language for maximum accuracy.

"""

        fix_prompt = """
Please fix the format of the output to be a valid Python dictionary with this structure:

{
  "search": list[list[str]],   # list of lists of strings
  "highlight": list[str]       # list of strings
}

Ensure:
- All elements are strings.
- No extra commentary.
- Keys must be "search" and "highlight".
- No trailing commas or syntax errors.

Return only the fixed dictionary.
"""

        l = getLogger()
        for attempt in range(self.max_retries):
            user_prompt = f"provide result for: '{user_query}'"
            l.info("user prompt:", user_prompt)
            response_text = self._query_chatgpt(system_prompt, user_prompt)
            parsed_search_highlight = self._try_parse_search_highlight_json(response_text)
            if parsed_search_highlight is not None:
                return parsed_search_highlight
            else:
                l.info(f"[Attempt {attempt+1}] Invalid SearchHighlight format. \n{response_text}\n Retrying with fix prompt.")
                user_query = f"Fix this output:\n{response_text}"
                system_prompt = fix_prompt
        l.info("Max retries reached. Failed to get valid SearchHighlight.")
        return None