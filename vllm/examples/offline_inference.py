from vllm import LLM, SamplingParams
import time
# Sample prompts.
import os


prompts = [
    "Hello "*1000,

]*1
print(f"prompts len {len(prompts)}")
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=1)

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",tensor_parallel_size=1,pipeline_parallel_size=1,worker_use_ray=False)

begin=time.monotonic()
outputs = llm.generate(prompts, sampling_params)
end=time.monotonic()
print("time cost:",end-begin)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
print("done")



question="How do you believe the increasing integration of artificial intelligence into everyday life will impact social connections and interpersonal relationships over the next decade, particularly considering the balance between technological convenience and authentic human interaction in different cultural contexts?"
question_2="In what ways do you think the global shift toward sustainable energy and the increasing artificial intelligence into everyday life will transform urban infrastructure, economic systems, and individual lifestyle choices over the next thirty years, and what potential challenges might emerge during this transition that could require unprecedented cooperation between governments, private industries, and local communities?"

refine_question_number=3

max_tokens=int(len(question.split(" "))*refine_question_number*1.5)
sampling_params = SamplingParams(temperature=0.9, top_p=0.95,max_tokens=max_tokens,support_partial_output=True)
print("max_tokens:",sampling_params.max_tokens)


prompt=f"please refine and make the prompt suitable for LLM: \
    Please rewrite the following question into {refine_question_number} more refined one. \
    You should keep the original meaning of the question, but make it more concise and clear. \
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

prompt_2=f"please refine and make the prompt suitable for LLM: \
    Please rewrite the following question into {refine_question_number} more refined one. \
    You should keep the original meaning of the question, but make it more concise and clear. \
    The original question is: {question_2}? \
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


print("prompt:",prompt) 

prompt=f"\
    You are a helpful assistant. You are given a context and a question. \
    The context 1 is: NBA is a professional basketball league in the United States. Kevin Durant is a basketball player in the NBA.  Kevin Wayne Durant (born September 29, 1988), also known by his initials KD, is an American professional basketball player for the Phoenix Suns of the National Basketball Association (NBA). Durant has won two NBA championships, four Olympic gold medals, an NBA Most Valuable Player Award, two Finals MVP Awards, two NBA All-Star Game Most Valuable Player Awards, four NBA scoring titles, and the NBA Rookie of the Year Award; been named to 11 All-NBA teams (including six First Teams); and selected 15 times as an NBA All-Star. In 2021, Durant was named to the NBA 75th Anniversary Team. He ranks eighth among NBA career scoring leaders. Durant was a heavily recruited high school prospect widely regarded as the second-best player in his class. He played one season of college basketball for the Texas Longhorns, where he won numerous year-end awards and became the first freshman to be named Naismith College Player of the Year. Durant was selected as the second overall pick by the Seattle SuperSonics in the 2007 NBA draft. He played nine seasons with the franchise (which became the Oklahoma City Thunder in 2008) and led them to a Finals appearance in 2012. He later signed with the Golden State Warriors in 2016, who won a record 73 regular season games the previous year. Durant won consecutive NBA championships and NBA Finals MVP Awards in 2017 and 2018. After sustaining an Achilles injury in the 2019 NBA Finals, Durant joined the Brooklyn Nets as a free agent that summer. Following disagreements with the Nets' front office, he requested a trade during the 2022 offseason and was traded to the Suns the following year.\
    The context 2 is: NBA is a professional basketball league in the United States. Kevin Durant is a basketball player in the NBA.  Kevin Wayne Durant (born September 29, 1988), also known by his initials KD, is an American professional basketball player for the Phoenix Suns of the National Basketball Association (NBA). Durant has won two NBA championships, four Olympic gold medals, an NBA Most Valuable Player Award, two Finals MVP Awards, two NBA All-Star Game Most Valuable Player Awards, four NBA scoring titles, and the NBA Rookie of the Year Award; been named to 11 All-NBA teams (including six First Teams); and selected 15 times as an NBA All-Star. In 2021, Durant was named to the NBA 75th Anniversary Team. He ranks eighth among NBA career scoring leaders. Durant was a heavily recruited high school prospect widely regarded as the second-best player in his class. He played one season of college basketball for the Texas Longhorns, where he won numerous year-end awards and became the first freshman to be named Naismith College Player of the Year. Durant was selected as the second overall pick by the Seattle SuperSonics in the 2007 NBA draft. He played nine seasons with the franchise (which became the Oklahoma City Thunder in 2008) and led them to a Finals appearance in 2012. He later signed with the Golden State Warriors in 2016, who won a record 73 regular season games the previous year. Durant won consecutive NBA championships and NBA Finals MVP Awards in 2017 and 2018. After sustaining an Achilles injury in the 2019 NBA Finals, Durant joined the Brooklyn Nets as a free agent that summer. Following disagreements with the Nets' front office, he requested a trade during the 2022 offseason and was traded to the Suns the following year.\
    The context 3 is: NBA is a professional basketball league in the United States. Kevin Durant is a basketball player in the NBA.  Kevin Wayne Durant (born September 29, 1988), also known by his initials KD, is an American professional basketball player for the Phoenix Suns of the National Basketball Association (NBA). Durant has won two NBA championships, four Olympic gold medals, an NBA Most Valuable Player Award, two Finals MVP Awards, two NBA All-Star Game Most Valuable Player Awards, four NBA scoring titles, and the NBA Rookie of the Year Award; been named to 11 All-NBA teams (including six First Teams); and selected 15 times as an NBA All-Star. In 2021, Durant was named to the NBA 75th Anniversary Team. He ranks eighth among NBA career scoring leaders. Durant was a heavily recruited high school prospect widely regarded as the second-best player in his class. He played one season of college basketball for the Texas Longhorns, where he won numerous year-end awards and became the first freshman to be named Naismith College Player of the Year. Durant was selected as the second overall pick by the Seattle SuperSonics in the 2007 NBA draft. He played nine seasons with the franchise (which became the Oklahoma City Thunder in 2008) and led them to a Finals appearance in 2012. He later signed with the Golden State Warriors in 2016, who won a record 73 regular season games the previous year. Durant won consecutive NBA championships and NBA Finals MVP Awards in 2017 and 2018. After sustaining an Achilles injury in the 2019 NBA Finals, Durant joined the Brooklyn Nets as a free agent that summer. Following disagreements with the Nets' front office, he requested a trade during the 2022 offseason and was traded to the Suns the following year.\
    The context 4 is: NBA is a professional basketball league in the United States. Kevin Durant is a basketball player in the NBA.  Kevin Wayne Durant (born September 29, 1988), also known by his initials KD, is an American professional basketball player for the Phoenix Suns of the National Basketball Association (NBA). Durant has won two NBA championships, four Olympic gold medals, an NBA Most Valuable Player Award, two Finals MVP Awards, two NBA All-Star Game Most Valuable Player Awards, four NBA scoring titles, and the NBA Rookie of the Year Award; been named to 11 All-NBA teams (including six First Teams); and selected 15 times as an NBA All-Star. In 2021, Durant was named to the NBA 75th Anniversary Team. He ranks eighth among NBA career scoring leaders. Durant was a heavily recruited high school prospect widely regarded as the second-best player in his class. He played one season of college basketball for the Texas Longhorns, where he won numerous year-end awards and became the first freshman to be named Naismith College Player of the Year. Durant was selected as the second overall pick by the Seattle SuperSonics in the 2007 NBA draft. He played nine seasons with the franchise (which became the Oklahoma City Thunder in 2008) and led them to a Finals appearance in 2012. He later signed with the Golden State Warriors in 2016, who won a record 73 regular season games the previous year. Durant won consecutive NBA championships and NBA Finals MVP Awards in 2017 and 2018. After sustaining an Achilles injury in the 2019 NBA Finals, Durant joined the Brooklyn Nets as a free agent that summer. Following disagreements with the Nets' front office, he requested a trade during the 2022 offseason and was traded to the Suns the following year.\
    The question is:  Which team picked Kevin Durant in the 2007 NBA draft? \
    Please answer the question based on the context. \
    Please output your answer directly, do not include any other information or additional text!!! \
    Always be concise and to the point. \
    The answer:"

begin=time.monotonic()
outputs=llm.generate([prompt], sampling_params)
end=time.monotonic()
print("output:",outputs[0].outputs[0].text)
print('output token length:',len(outputs[0].outputs[0].token_ids))
print('output token:',outputs)
print("time cost:",end-begin)

