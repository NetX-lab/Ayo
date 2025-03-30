from vllm import LLM, SamplingParams
import time
import os

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",tensor_parallel_size=1,pipeline_parallel_size=1,worker_use_ray=False)


question="How do you believe the increasing integration of artificial intelligence into everyday life will impact social connections and interpersonal relationships over the next decade, particularly considering the balance between technological convenience and authentic human interaction in different cultural contexts?"

refine_question_number=5
max_tokens=int(len(question.split(" "))*refine_question_number*1.5)
sampling_params = SamplingParams(temperature=0.9, top_p=0.95,max_tokens=max_tokens,support_partial_output=True)
print("max_tokens:",sampling_params.max_tokens)



prompt=f"please refine and make the prompt suitable for LLM: \
    Please rewrite the following question into {refine_question_number} more refined one. \
    You should keep the original meaning of the question, but make it more detailed and clear. \
    The original question is: {question}? \
    Please output your answer in json format. \
    It should contain {refine_question_number} questions, and the keys are " + \
    ", ".join([f"question{i+1}" for i in range(refine_question_number)]) + \
    ", and the values are the refined questions. \
    Here is a reference json output: \
    {{ \
" + "\n      ".join([f"\"question{i+1}\": \"[refined version {i+1}]\"" + ("," if i < refine_question_number-1 else "") for i in range(refine_question_number)]) + " \
    }} \
    You just need to output the json string, do not output any other information or additional text!!! \
    The json output:"

prompt=f"\
    You are a helpful assistant. You are given a context and a question. \
    The context 1 is: NBA is a professional basketball league in the United States. Kevin Durant is a basketball player in the NBA.  Kevin Wayne Durant (born September 29, 1988), also known by his initials KD, is an American professional basketball player for the Phoenix Suns of the National Basketball Association (NBA). Durant has won two NBA championships, four Olympic gold medals, an NBA Most Valuable Player Award, two Finals MVP Awards, two NBA All-Star Game Most Valuable Player Awards, four NBA scoring titles, and the NBA Rookie of the Year Award; been named to 11 All-NBA teams (including six First Teams); and selected 15 times as an NBA All-Star. In 2021, Durant was named to the NBA 75th Anniversary Team. He ranks eighth among NBA career scoring leaders. Durant was a heavily recruited high school prospect widely regarded as the second-best player in his class. He played one season of college basketball for the Texas Longhorns, where he won numerous year-end awards and became the first freshman to be named Naismith College Player of the Year. Durant was selected as the second overall pick by the Seattle SuperSonics in the 2007 NBA draft. He played nine seasons with the franchise (which became the Oklahoma City Thunder in 2008) and led them to a Finals appearance in 2012. He later signed with the Golden State Warriors in 2016, who won a record 73 regular season games the previous year. Durant won consecutive NBA championships and NBA Finals MVP Awards in 2017 and 2018. After sustaining an Achilles injury in the 2019 NBA Finals, Durant joined the Brooklyn Nets as a free agent that summer. Following disagreements with the Nets' front office, he requested a trade during the 2022 offseason and was traded to the Suns the following year.\
    The context 2 is: NBA is a professional basketball league in the United States. Kevin Durant is a basketball player in the NBA.  Kevin Wayne Durant (born September 29, 1988), also known by his initials KD, is an American professional basketball player for the Phoenix Suns of the National Basketball Association (NBA). Durant has won two NBA championships, four Olympic gold medals, an NBA Most Valuable Player Award, two Finals MVP Awards, two NBA All-Star Game Most Valuable Player Awards, four NBA scoring titles, and the NBA Rookie of the Year Award; been named to 11 All-NBA teams (including six First Teams); and selected 15 times as an NBA All-Star. In 2021, Durant was named to the NBA 75th Anniversary Team. He ranks eighth among NBA career scoring leaders. Durant was a heavily recruited high school prospect widely regarded as the second-best player in his class. He played one season of college basketball for the Texas Longhorns, where he won numerous year-end awards and became the first freshman to be named Naismith College Player of the Year. Durant was selected as the second overall pick by the Seattle SuperSonics in the 2007 NBA draft. He played nine seasons with the franchise (which became the Oklahoma City Thunder in 2008) and led them to a Finals appearance in 2012. He later signed with the Golden State Warriors in 2016, who won a record 73 regular season games the previous year. Durant won consecutive NBA championships and NBA Finals MVP Awards in 2017 and 2018. After sustaining an Achilles injury in the 2019 NBA Finals, Durant joined the Brooklyn Nets as a free agent that summer. Following disagreements with the Nets' front office, he requested a trade during the 2022 offseason and was traded to the Suns the following year.\
    The question is:  Which team picked Kevin Durant in the 2007 NBA draft? \
    Please answer the question based on the context. \
    Please output your answer directly, do not include any other information or additional text!!! \
    Always be concise and to the point. \
    The answer:"

prompt_words=prompt.split(" ")
prompt_words_length=len(prompt_words)

prompt_part_1=prompt_words[:int(prompt_words_length*0.8)]
prompt_part_2=prompt_words[int(prompt_words_length*0.8):]


prompt_part_1_str=" ".join(prompt_part_1)
prompt_part_2_str=" ".join(prompt_part_2)

print("prompt_part_1_str:",prompt_part_1_str)
print("prompt_part_2_str:",prompt_part_2_str)

llm.llm_engine.add_request(request_id="warmup_request",prompt=prompt,sampling_params=sampling_params,partial_prefilling=False)

while llm.llm_engine.has_unfinished_requests():
    step_outputs = llm.llm_engine.step()
    for output in step_outputs:
        print("warm up output:",output)

llm.llm_engine.add_request(request_id="test_partial_prefilling",prompt=prompt_part_1_str,sampling_params=sampling_params,partial_prefilling=True)

begin=time.monotonic()

add_prompt_chunk_2 = False

outputs=[]
step_times=[]

while llm.llm_engine.has_unfinished_requests():
     
    begin_step=time.monotonic()
    step_outputs = llm.llm_engine.step()
    for output in step_outputs:

        if output.finished:
            outputs.append(output)

    if not add_prompt_chunk_2:
        llm.llm_engine.add_prompt_chunk_for_existing_request(request_id="test_partial_prefilling",prompt_chunk_str=prompt_part_2_str,prompt_chunk_token_ids=llm.get_tokenizer().encode(prompt_part_2_str),is_last_chunk=True)
        add_prompt_chunk_2 = True
    end_step=time.monotonic()
    print("step time cost:",end_step-begin_step)
    step_times.append(end_step-begin_step)

end=time.monotonic()
print("time cost:",end-begin)

print("outputs:",outputs)

#the step times for partial prefilling, full prefilling, and several decoding steps
print("step_times:",step_times)



