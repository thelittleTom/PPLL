
class Statistics:
    def __init__(self):
        self.cost=0.0
        self.time=None
    def __str__(self):
        return (f"----statistics-----\n"
                f"cost:{self.cost}\n"
                f"time:{self.time}\n"
                f"usage:{self.usage}\n"
                f"-------------------\n")

from langchain_openai import AzureChatOpenAI


llm_35_stable=   AzureChatOpenAI(
            deployment_name="gpt35",
            api_key="",
            api_version="",
            azure_endpoint="",
            temperature=0.00000001,
        )


from openai import AzureOpenAI


from openai import OpenAI

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

    llm=llm_35_stable

    with get_openai_callback() as cb:
        out = llm.invoke([message]).content
        cost= cb.total_cost
        sta.cost += cost

        #print(
            #f"Total Cost (USD): ${format( cost, '.6f')}"
        #)  # without specifying the model version, flat-rate 0.002 USD per 1k input and output tokens is used

    # print(out)

    return out

