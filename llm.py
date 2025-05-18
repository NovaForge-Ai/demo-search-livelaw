import openai
import ast
from typing import List, Tuple, Union
import os

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
        system_prompt: str,
        user_query: str,
        fix_array_system_prompt: str
    ) -> Union[List[Tuple[str, int]], None]:
        """Main method to fetch an array of tuples from ChatGPT."""
        for attempt in range(self.max_retries):
            response_text = self._query_chatgpt(system_prompt, f"provide result for: '{user_query}'")
            parsed_array = self._try_parse_tuple_array(response_text)
            if parsed_array is not None:
                return parsed_array
            else:
                print(f"[Attempt {attempt+1}] Invalid tuple array format. Retrying with fix prompt.")
                user_query = f"Fix this output:\n{response_text}"
                system_prompt = fix_array_system_prompt
        print("Max retries reached. Failed to get valid tuple array.")
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
        return response.choices[0].message.content.strip()

    def _try_parse_tuple_array(self, output: str) -> Union[List[Tuple], None]:
        """Try to parse the output as a list of tuples using ast.literal_eval."""
        try:
            output = output.replace("\n", "")
            parsed = ast.literal_eval(output)
            if isinstance(parsed, list) and all(isinstance(item, tuple) and isinstance(item[0], str) and isinstance(item[1], int) for item in parsed):
                return parsed
            else:
                print("Parsed result is not a list of tuples.")
                return None
        except Exception as e:
            print(f"Failed to parse output as Python list of tuples: {e}")
            return None