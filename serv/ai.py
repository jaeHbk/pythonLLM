import os
import openai
from dotenv import load_dotenv
load_dotenv()


def generate_review(review, qa):

    print(review)
    res = qa.run(review)
    print(res)
    return res
    