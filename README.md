# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

### Task 1: Chain Design Visualization & Solution Description

```mermaid
graph TD
    A[Receipt Images Folder] --> B["LangChain .batch() Processing"]
    B --> C[ChatPromptTemplate]
    C -->|"Q1: Final Payment & Q2: Subtotal + Discounts"| D[ChatDeepSeek]
    D -->|deepseek-v4-flash-vision-exp| E[StrOutputParser]
    E --> F[Text Outputs]
    F --> G[Regex Extraction in Python]
    G --> H[Decimal Aggregation]
    H --> I[Final Answer Dictionary]
```

**Solution Description:**
To ensure computational accuracy and bypass the inherent limitations of Large Language Models in performing complex floating-point arithmetic across multiple images, I designed a robust two-stage processing pipeline. First, the LangChain chain utilizes a `ChatPromptTemplate` to instruct the `deepseek-v4-flash-vision-exp` vision model to independently extract exactly two targeted values from each receipt image. I leveraged LangChain's `.batch()` method to process all image inputs in parallel, significantly optimizing the execution speed. In the second stage, instead of relying on the LLM to sum the values, I implemented a Python-based aggregation step using regular expressions (`re`) to parse the model's text outputs. The extracted numbers are instantly converted into `Decimal` objects, ensuring strict precision during the final sum calculation across all receipts, thereby eliminating floating-point errors and mathematical hallucinations.

---

