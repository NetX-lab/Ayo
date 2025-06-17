import re

QUERY_EXPANDING_PROMPT_TEMPLATE_STRING = """\
        Please rewrite the following question into {refine_question_number} more refined one. \
        You should keep the original meaning of the question, but make it more suitable and clear for context retrieval. \
        The original question is: {question}? \
        Please output your answer in json format. \
        It should contain {refine_question_number} new refined questions.\
        For example, if the expaned number is 3, the json output should be like this: \
        {{\
            "revised question1": "[refined question 1]",\
            "revised question2": "[refined question 2]",\
            "revised question3": "[refined question 3]"\
        }}\
        You just need to output the json string, do not output any other information or additional text!!! \
        The json output:"""

RAG_QUESTION_ANSWERING_PROMPT_TEMPLATE_STRING = """\
      You are an AI assistant specialized in Retrieval-Augmented Generation (RAG). Your responses
      must be based strictly on the retrieved documents provided to you. Follow these guidelines:
      1. Use Retrieved Information Only - Your responses must rely solely on the retrieved documents.
      If the retrieved documents do not contain relevant information, explicitly state: 'Based on the
      available information, I cannot determine the answer.'\n"
      2. Response Formatting - Directly answer the question using the retrieved data. If multiple
      sources provide information, synthesize them in a coherent manner. If no relevant information
      is found, clearly state that.\n"
      3. Clarity and Precision - Avoid speculative language such as 'I think' or 'It might be.'
      Maintain a neutral and factual tone.\n"
      4. Information Transparency - Do not fabricate facts or sources. If needed, summarize the
      retrieved information concisely.\n"
      5. Handling Out-of-Scope Queries - If a question is outside the retrieved data (e.g., opinions,
      unverifiable claims), state: 'The retrieved documents do not provide information on this topic.'\n
      ---\n
      Example Interactions:\n
      User Question: Who founded Apple Inc.?\n
      Retrieved Context: 'Apple Inc. was co-founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.'\n
      Model Answer: 'Apple Inc. was co-founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.'\n
      ---\n
      User Question: When was the first iPhone released, and what were its key features?\n"
      Retrieved Context: 'The first iPhone was announced by Steve Jobs on January 9, 2007, and released on June 29, 2007.' "
      "'The original iPhone featured a 3.5-inch touchscreen display, a 2-megapixel camera, and ran on iOS.'\n"
      Model Answer: 'The first iPhone was announced on January 9, 2007, and released on June 29, 2007. "
      "It featured a 3.5-inch touchscreen display, a 2-megapixel camera, and ran on iOS.'\
      This ensures accuracy, reliability, and transparency in all responses. And you should directly answer the question based on the retrieved context and keep it concise as possible.
      Here is the question: {question}?
      Here is the retrieved context: {context}
      Here is your answer, make sure it is concise:
     """


def replace_placeholders(prompt_template: str, **kwargs):
    for key, value in kwargs.items():
        prompt_template = prompt_template.replace(f"{{{key}}}", f"{{{value}}}")
    print(prompt_template)
    return prompt_template


# TODO: Currently, these classes have been actually used in the modules and the related payload-transformations
# We should do these to make the prompt-template-transformation more flexible and reusable.


class PromptTemplate:
    def __init__(self, prompt_template: str):
        # the template is a string with placeholders like {key}
        self.prompt_template = prompt_template

        self.placeholders = re.findall(r"\{([^{}]+)\}", self.prompt_template)

    def fill_template(self, **kwargs):
        raise NotImplementedError("This method should be implemented by the subclass")


class QueryExpandingPromptTemplate(PromptTemplate):

    default_template = """\
        Please rewrite the following question into {refine_question_number} more refined one. \
        You should keep the original meaning of the question, but make it more suitable and clear for context retrieval. \
        The original question is: {question}? \
        Please output your answer in json format. \
        It should contain {refine_question_number} new refined questions.\
        For example, if the expaned number is 3, the json output should be like this: \
        {{\
            "revised question1": "[refined question 1]",\
            "revised question2": "[refined question 2]",\
            "revised question3": "[refined question 3]"\
        }}\
        You just need to output the json string, do not output any other information or additional text!!! \
        The json output:"""

    def __init__(self, prompt_template: str = None):
        if prompt_template is None:
            prompt_template = self.default_template
        super().__init__(prompt_template)

        # special case, here we do not need to check the placeholders
        self.placeholders = re.findall(r"\{([^{}]+)\}", self.prompt_template)

    def fill_template(self, **kwargs):
        refine_question_number = None
        question = None
        for key, value in kwargs.items():
            if "num" in key.lower():
                refine_question_number = value
            elif key.lower() in ["query", "question", "question_"]:
                question = value

        assert refine_question_number is not None, "refine_question_number is required"
        assert question is not None, "question is required"

        keys = ", ".join([f"question{i+1}" for i in range(refine_question_number)])
        json_example = (
            "{\n      "
            + "\n      ".join(
                [
                    f'"question{i+1}": "[refined version {i+1}]"'
                    + ("," if i < refine_question_number - 1 else "")
                    for i in range(refine_question_number)
                ]
            )
            + "\n    }"
        )

        return self.prompt_template.format(
            refine_question_number=refine_question_number,
            question=question,
            keys=keys,
            json_example=json_example,
        )


class RAGQuestionAnsweringPromptTemplate(PromptTemplate):

    default_template = """\
      You are an AI assistant specialized in Retrieval-Augmented Generation (RAG). Your responses
      must be based strictly on the retrieved documents provided to you. Follow these guidelines:
      1. Use Retrieved Information Only - Your responses must rely solely on the retrieved documents.
      If the retrieved documents do not contain relevant information, explicitly state: 'Based on the
      available information, I cannot determine the answer.'\n"
      2. Response Formatting - Directly answer the question using the retrieved data. If multiple
      sources provide information, synthesize them in a coherent manner. If no relevant information
      is found, clearly state that.\n"
      3. Clarity and Precision - Avoid speculative language such as 'I think' or 'It might be.'
      Maintain a neutral and factual tone.\n"
      4. Information Transparency - Do not fabricate facts or sources. If needed, summarize the
      retrieved information concisely.\n"
      5. Handling Out-of-Scope Queries - If a question is outside the retrieved data (e.g., opinions,
      unverifiable claims), state: 'The retrieved documents do not provide information on this topic.'\n
      ---\n
      Example Interactions:\n
      User Question: Who founded Apple Inc.?\n
      Retrieved Context: 'Apple Inc. was co-founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.'\n
      Model Answer: 'Apple Inc. was co-founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.'\n
      ---\n
      User Question: When was the first iPhone released, and what were its key features?\n"
      Retrieved Context: 'The first iPhone was announced by Steve Jobs on January 9, 2007, and released on June 29, 2007.' "
      "'The original iPhone featured a 3.5-inch touchscreen display, a 2-megapixel camera, and ran on iOS.'\n"
      Model Answer: 'The first iPhone was announced on January 9, 2007, and released on June 29, 2007. "
      "It featured a 3.5-inch touchscreen display, a 2-megapixel camera, and ran on iOS.'\
      This ensures accuracy, reliability, and transparency in all responses. And you should directly answer the question based on the retrieved context and keep it concise as possible.
      Here is the question: {question}?
      Here is the retrieved context: {context}
      Here is your answer:
     """

    def __init__(self, prompt_template: str = None):

        if prompt_template is None:
            prompt_template = self.default_template
        super().__init__(prompt_template)


# TODO: add more prompt templates
