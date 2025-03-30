from typing import List
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeType, NodeOps, NodeIOSchema
from Ayo.engines.engine_types import EngineType
from Ayo.modules.base_module import BaseModule
from enum import Enum

class LLMGenerationMode(Enum):
    NORMAL = "normal"
    SUMMARIZATION = "summarization"
    REFINEMENT = "refinement"

class LLMSynthesizingModule(BaseModule):
   RAG_PROMPT_TEMPLATE = """\
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

   SUMMARIZATION_PROMPT_TEMPLATE = """\
      You are an AI assistant specialized in Question Answering. You would be provided with a question and several candidate answers. 
      Your task is to summarize the candidate answers into a single answer. You should keep the original meaning of the question and the candidate answers.
      You should select the most relevant answer from the candidate answers and summarize it. Always need to keep your answer concise and to the point.
      Here is the question: {question}?  
      Here are the candidate answers: {answers}
      Here is your answer:
     """ 
   
   def __init__(self,
                input_format: dict={
                   "question": str,
                   "context": List[str]
                },
                output_format: dict={
                   "answer": str},
                config: dict={
                    "generation_mode":LLMGenerationMode.NORMAL,
                    'prompt_template': RAG_PROMPT_TEMPLATE,
                    'parse_json': True, 
                    'prompt':RAG_PROMPT_TEMPLATE,
                    'partial_output': False,
                    'partial_prefilling': False,
                    'llm_partial_decoding_idx': -1
                }):
      """Initialize the LLM Synthesizing Module.
      
      This module is responsible for generating answers using a Large Language Model (LLM)
      based on retrieved context. It supports multiple generation modes: normal generation,
      summarization, and refinement.
      
      Args:
          input_format (dict): Input format definition, defaults to:
              - question (str): User question
              - context (List[str]): List of retrieved context documents
          output_format (dict): Output format definition, defaults to:
              - answer (str): Generated response
          config (dict): Configuration parameters, including:
              - generation_mode (LLMGenerationMode): Generation mode, can be NORMAL, SUMMARIZATION, or REFINEMENT
              - prompt_template (str): Prompt template string
              - parse_json (bool): Whether to parse JSON output
              - prompt (str): Complete prompt string
              - partial_output (bool): Whether to enable partial output
              - partial_prefilling (bool): Whether to enable partial prefilling
              - llm_partial_decoding_idx (int): Partial decoding index
              
      Notes:
          - Summarization mode (SUMMARIZATION) requires 'context_num' to be specified in config
          - Refinement mode (REFINEMENT) also requires 'context_num' to be specified in config
      """
      
      super().__init__(input_format, output_format, config)

      # TODO: support the below generation mode 
      if config["generation_mode"]==LLMGenerationMode.SUMMARIZATION:
         assert 'context_num' in config, "context_num is required for summarization"
      elif config["generation_mode"]==LLMGenerationMode.REFINEMENT:
         assert 'context_num' in config, "context_num is required for refinement"


   def to_primitive_nodes(self):
      if self.config["generation_mode"]==LLMGenerationMode.NORMAL: 
         return [
            Node(
                name="LLMSynthesizingPrefilling",
                input_format=self.input_format,
                output_format=self.output_format,
                node_type=NodeType.COMPUTE,
                engine_type=EngineType.LLM,
                op_type=NodeOps.LLM_PREFILLING,
                config=self.config
            ),
            Node(
                name="LLMSynthesizingDecoding",
                input_format=self.input_format,
                output_format=self.output_format,
                node_type=NodeType.COMPUTE,
                engine_type=EngineType.LLM,
                op_type=NodeOps.LLM_DECODING,
                config=self.config
            )
         ]
      
      elif self.config["generation_mode"]==LLMGenerationMode.SUMMARIZATION:
         return [
            Node(
                name="LLMSynthesizingPrefilling",
            )
         ]
      
      elif self.config["generation_mode"]==LLMGenerationMode.REFINEMENT:
         return [
            Node(
                name="LLMSynthesizingPrefilling",
            )
         ] 

