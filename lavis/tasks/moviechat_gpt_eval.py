import ast
import json
import multiprocessing
import os
import time

import openai
from openai.error import APIError, RateLimitError, Timeout
from tqdm import tqdm

api_key = os.getenv("OPENAI_API_KEY")


def evaluate_qa_pair(data, retries=3, delay=10):
    question, answer, pred = data["question"], data["answer"], data["pred"]
    prompt = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    user_input = (
        f"Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "Your response should look strictly follow this format: {'pred': 'yes', 'score': 4}."
    )

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input},
                ],
                timeout=30,  # Set a timeout for the API call
            )
            response_message = response["choices"][0]["message"]["content"]
            return ast.literal_eval(response_message)
        except (RateLimitError, APIError) as e:
            print(
                f"Rate limit or API error encountered: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
        except Timeout as e:
            print(f"Request timed out: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None
    # The rest of the function remains the same as before, just use the variables directly


def worker(data):
    return evaluate_qa_pair(data)


def main(pred_path):
    # with open(pred_path, 'r') as pred_file:
    pred_contents = pred_path

    data_for_workers = []
    for sample in pred_contents:
        id = sample.split(".")[0]
        qa_list = pred_contents[sample]
        for qa_pair in qa_list:
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            pred = qa_pair["pred"].replace("</s>", "")
            data_for_workers.append(
                {"question": question, "answer": answer, "pred": pred}
            )

    total_items = len(data_for_workers)

    pool = multiprocessing.Pool(processes=8)

    with tqdm(total=total_items, desc="Processing") as pbar:
        results = []
        for result in pool.imap_unordered(worker, data_for_workers):
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    yes_count = 0
    no_count = 0
    score_sum = 0

    for result in results:
        if result:
            try:
                if result["pred"].lower() == "yes":
                    yes_count += 1
                else:
                    no_count += 1
                score_sum += result["score"]
            except:
                print("Error in result: ", result)
                continue

    average_score = score_sum / (yes_count + no_count) if (yes_count + no_count) else 0
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) else 0
    return {"accuracy": accuracy * 100.0, "average_score": average_score}


if __name__ == "__main__":
    pred_path = "moviechat_gpt_eval.json"
    res = main(pred_path)
    print(res)
