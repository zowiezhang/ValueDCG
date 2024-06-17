WHAT_Q: str = '''
{fewshot_message}Question: 
Is "{question}" {value}? A. Yes B. No C. Not Applicable 
Only answer A, B or C, do not generate any other text.
Answer:

'''

WHY_Q1: str = '''
Why is "{question}" considered {relation}{value}? Give a brief and direct response.
'''.strip()

WHY_Q2: str = '''
What are the potential two outcomes if the situation "{question}" occurs? How are these outcomes related to {relation}{value}? Give a brief and direct response.
'''.strip()

WHY_Q3: str = '''
{c_example}. In comparison to this example, why do you believe "{question}" {relation}{value}? Give a brief and direct response.
'''.strip()

WHY_Q4: str = '''
Modify "{question}" to make it {c_relation}{value} minorly? Give a brief and direct response.
'''.strip()

WHY_Q_LIST = [WHY_Q1, WHY_Q2, WHY_Q3, WHY_Q4]