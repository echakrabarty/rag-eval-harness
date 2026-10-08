"""Prompt rendering + generators.

The default generator is extractive (no API key needed) so the harness runs
offline end-to-end. Swap in a real LLM by implementing the same interface.
"""


def render_prompt(template_path, question, contexts):
    with open(template_path) as f:
        template = f.read()
    context_block = "\n\n".join(f"[{c['id']}] {c['text']}" for c in contexts)
    return template.replace("{{question}}", question).replace("{{context}}", context_block)


class ExtractiveGenerator:
    """Mock generator: answers with the top-ranked chunk, cited.

    Useful for smoke-testing the harness. Replace with an LLM call for real evals.
    """

    def generate(self, prompt, contexts):
        top = contexts[0]
        return f"Based on [{top['id']}]: {top['text']}"


# Example: plug in a real LLM (OpenAI-compatible client)
#
# class LLMGenerator:
#     def __init__(self, client, model="gpt-4o-mini"):
#         self.client = client
#         self.model = model
#
#     def generate(self, prompt, contexts):
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#         )
#         return resp.choices[0].message.content
