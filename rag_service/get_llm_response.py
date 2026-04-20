import http.client
import json
import time
from typing import Any
import traceback
from retrieval.retriever import RAGInterface
from prompt_template import auto_build_prompt

class HttpsApi():
    def __init__(self, host, key, model, timeout=20, **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 4096),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                # print(data)
                response = data['choices'][0]['message']['content']
                return response
            except Exception as e:

                print(f'Error: {traceback.format_exc()}.'
                          f'You may check your API host and API key.')
                time.sleep(2)
                continue


if __name__ == "__main__":
    host = 'api.bltcy.top'
    key = 'sk-Clt5fdhN9xAT9sk2aj6MRCEgF8Zv7ahy3KQP1RK5PqHRGpCP'
    model = 'gpt-4o-mini-2024-07-18'
    http_client = HttpsApi(host=host, key=key, model=model)
    rag = RAGInterface()

    query = "意外医疗保险如何理赔？"
    results = rag.retrieve(query, top_k=3)
    ref_text = []
    for i, r in enumerate(results, 1):
        # print(f"{i}. [score={r['score']}] {r['text'][:200]} ...")
        ref_text.append(r['text']+'\n')

    # 选择风格模板（'expert'、'customer'、'academic'、'json'）
    mode = "expert"

    # 自动判断语言，生成对应语言 prompt
    prompt = auto_build_prompt(query, ref_text, mode=mode)
    response = http_client.draw_sample(prompt=prompt)

    print(response)
