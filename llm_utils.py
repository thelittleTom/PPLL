
class Statistics:
    def __init__(self):
        self.cost=0.0
        self.usage={'gpt4':0,'gpt4o':0 }
        self.time=None
    def __str__(self):
        return (f"----statistics-----\n"
                f"cost:{self.cost}\n"
                f"time:{self.time}\n"
                f"usage:{self.usage}\n"
                f"-------------------\n")

from langchain_openai import AzureChatOpenAI

llm_4o_diverse = AzureChatOpenAI(
    deployment_name="gpt-4o",
    api_key="afcdd8c90afc40cfabdc89a096fdaab1",
    api_version="2023-10-01-preview",
    azure_endpoint="https://chatgpt-nc-us.openai.azure.com/",
            temperature=0.9,
        )
llm_4o_stable = AzureChatOpenAI(
    deployment_name="gpt-4o",
    api_key="afcdd8c90afc40cfabdc89a096fdaab1",
    api_version="2023-10-01-preview",
    azure_endpoint="https://chatgpt-nc-us.openai.azure.com/",
            temperature=0.00000001,
        )
llm_35_diverse=   AzureChatOpenAI(
            deployment_name="gpt35-zhibo",
            api_key="3b962d3f14854d66856e22e1ba4f9c62",
            api_version="2023-10-01-preview",
            azure_endpoint="https://chatgpt-4-yiwise-ca.openai.azure.com/",
            temperature=0.4,
        )
llm_35_stable=   AzureChatOpenAI(
            deployment_name="gpt35-zhibo",
            api_key="3b962d3f14854d66856e22e1ba4f9c62",
            api_version="2023-10-01-preview",
            azure_endpoint="https://chatgpt-4-yiwise-ca.openai.azure.com/",
            temperature=0.00000001,
        )

from openai import AzureOpenAI
embeddingAI =    AzureOpenAI(
            api_key="3b962d3f14854d66856e22e1ba4f9c62",
            api_version="2023-10-01-preview",
            azure_endpoint="https://chatgpt-4-yiwise-ca.openai.azure.com/",
        )
def get_embedding(texts):
    rst=embeddingAI.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"
    )

    return rst.data[0].embedding

from openai import OpenAI
llm_deepSeek=  OpenAI(api_key="sk-a0e6a5512960442bb4f0ac805fbec1dc", base_url="https://api.deepseek.com")

from langchain_core.messages import HumanMessage

from langchain_community.callbacks import get_openai_callback

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
before_log,
after_log
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

sta=Statistics()
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(prompt_template,sta=sta,llm='llm_4_stable',model=None,tokenizer=None):
    if 'llm' not in llm:
        messages = [
            {"role": "system", "content": "You are a linguist who is good at clustering."},
            {"role": "user", "content": prompt_template},
        ]
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 4096,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args,pad_token_id=tokenizer.eos_token_id)
        return output[0]['generated_text']

    message = HumanMessage(content=prompt_template)
    if llm == 'llm_4o_diverse':
        llm = llm_4o_diverse
    elif llm=='llm_4o_stable':
        llm = llm_4o_stable
    elif llm=='llm_35_stable':
        llm=llm_35_stable
    elif llm=='llm_35_diverse':
        print('llm_35_d')
        llm=llm_35_diverse
    with get_openai_callback() as cb:
        out = llm.invoke([message]).content
        cost= cb.total_cost
        if llm.deployment_name == "gpt-4o":
            cost = cb.prompt_tokens * 0.005 / 1000 + cb.completion_tokens * 0.015 / 1000

        sta.cost += cost
        if llm.deployment_name == "gpt4-zhibo":
            sta.usage['gpt4'] += 1
        elif llm.deployment_name == "gpt-4o":
            sta.usage['gpt4o'] += 1
        print(
            f"Total Cost (USD): ${format( cost, '.6f')}"
        )  # without specifying the model version, flat-rate 0.002 USD per 1k input and output tokens is used

    # print(out)

    return out

