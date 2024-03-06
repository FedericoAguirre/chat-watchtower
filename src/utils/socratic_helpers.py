from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

def parse_socratic_parsing(explanation):
    prompt = PromptTemplate.from_template("""
    You are a world class text analyzer that can help in parsing an given
    INPUT. INPUT is provided after ### Your job is to parse the total number of questions that were answered "Yes"
    and the total number of questions present. Return the output in a JSON format with following keys. The
    description of each key below is given next to it. - socratic_explanations - List containing all the answers to
    the questions - socratic_results - List containing values "Yes" or "No". Each element in the list corresponds to
    one question in the input. "Yes" is choosen when the question was answered Yes else No.

    Place the above presented JSON formatted answer in the <output></output> tags.
    Look at the following examples on how the output has to be formatted.

    Example 1:
    INPUT:
    1. Does the answer mention any specific circumstances under which an employee can self-register a username using a foreign telephone number?
    - Yes, The answer mentions that Flex allows an employee to register with a foreign telephone number if they also use a foreign address.

    2. Does the answer state that the user's account setup must match exactly with the foreign address and phone number in order to register using a foreign telephone number?
    - No, The answer does not mention anything about the user's account setup needing to match exactly with the foreign address and phone number.

    3. Does the answer mention it is possible to self-register with a foreign telephone number?
    - Yes, The answer mentions that Flex allows an employee to register with a foreign telephone number.

    Total Number of Questions: 3
    Total questions answered with Yes: 2

    OUTPUT: <output>{{"socratic_explanations": ["Yes, The answer mentions that Flex allows an employee to register with a foreign telephone number if they also use a foreign address.",
    "No, The answer does not mention anything about the user's account setup needing to match exactly with the foreign address and phone number.",
    "Yes, The answer mentions that Flex allows an employee to register with a foreign telephone number."],
     "socratic_results": ["Yes", "No", "Yes"]}}</output>


    Example 2:
    INPUT:
    1. Does the answer mention about use of Splunk queries to search for data in a specific index?
    - The answer does not mention anything about using Splunk queries to search for data in a specific index.

    2. Does the answer explain how to specify which fields to include in the query result set?
    - The answer does not explain how to specify which fields to include in the query result set.

    3. Does the answer provide an example of a basic Splunk query and explain how it works?
    - The answer does not provide an example of a basic Splunk query and explain how it works.

    4. Does the answer contain a basic Splunk query?
    - The answer does not contain a basic Splunk query.

    OUTPUT: <output>{{"socratic_explanations": ["The answer does not mention anything about using Splunk queries to search for data in a specific index.",
    "The answer does not explain how to specify which fields to include in the query result set.",
    "The answer does not provide an example of a basic Splunk query and explain how it works.",
    "The answer does not contain a basic Splunk query."], "socratic_results": ["No", "No", "No", "No"]}}</output>

    Example 3:
    INPUT:
    1. Does the answer mention about Configuration > Payroll Policies > Timecard Pay Policies > Create New Policy as the location to set up a Flex Time payroll policy?
    - NO

    2. Does the answer mention about employees must have worked for at least 90 days before they are eligible for holiday pay?
    - YES

    3. Does the answer mention about the Qualifying Period can be set to 90 days?
    - YES

    4. Does the answer mention about the policy will only pay eligible employees for holidays based on qualifying criteria?
    - YES

    5. Does the answer mention about the policy will automatically be applied to all employees in the company?
    - NO

    Total Number of Questions: 5
    Total questions answered with Yes: 3

    OUTPUT: <output>{{"socratic_explanations": ["No, The answer does not mention about Configuration > Payroll Policies > Timecard Pay Policies > Create New Policy as the location to set up a Flex Time payroll policy?",
    "Yes, the answer does mention about employees must have worked for at least 90 days before they are eligible for holiday pay.",
    "Yes, the answer mentions about the Qualifying Period can be set to 90 days",
    "Yes,  the answer mention about the policy will only pay eligible employees for holidays based on qualifying criteria",
    "No, the answer does not mention about the policy will automatically be applied to all employees in the company"],
    "socratic_results": ["No", "Yes", "Yes", "Yes", "No"]}}</output>

    Example 4:
    INPUT:
    1. Does the answer mention that US Department of Labor website provides information about minimum wage requirements in a particular state?
    - Yes, the answer mentions that you can find information about minimum wage requirements in a particular state by visiting the U.S. Department of Labor's Wage and Hour Division website.

    2. Does the answer mention that latest minimum wage requirements for each state should be determined by using external websites?
    - No, the answer does not mention that the latest minimum wage requirements for each state should be determined by using external websites.

    3. Does the answer mention about cities or counties have higher minimum wage rates than the state minimum wage requirements?
    - No, the answer does not mention about cities or counties having higher minimum wage rates than the state minimum wage requirements.

    OUTPUT: <output>{{"socratic_explanations": ["Yes, the answer mentions that you can find information about minimum wage requirements in a particular state by visiting the U.S. Department of Labor's Wage and Hour Division website.",
    "No, the answer does not mention that the latest minimum wage requirements for each state should be determined by using external websites.",
    "No, the answer does not mention about cities or counties having higher minimum wage rates than the state minimum wage requirements."], "socratic_results": ["Yes", "No", "No"]}}</output>

    Now read the below INPUT and respond accordingly.
    ###
    INPUT: {input}
    OUTPUT:

    """,)
    model = OpenAI(api_key=openai_api_key, temperature=0)
    output = model(prompt.format(input=explanation))
    return output

socratic_prompt = """
            You are a world-class researcher that tests presence of required facts in an answer
            You are assessing a submitted answer on a given task based on a criterion. You must decide if the information the 
            Questions are looking for is in the CONTEXT.
            
            Here is the data:

            Answer the questions given after [BEGIN DATA] tags from the [CONTEXT] provided after [BEGIN DATA].
            Answer in YES or NO only. Provide only the answers as a bullet points of "YES" or "NO".
            Please repeat each question that was asked followed by "YES" or "NO" answer.
            "YES" followed by the explanation would be given in the cases where the information referenced in the Question is present in the Context.
            "NO" followed by the explanation in the cases where the information referenced in the Question is NOT present in the context. 
            Ignore the upper and lower case in answer.

            Check the following examples on how a set of Questions are answered.
            After looking at the examples, look at the Questions after ### and CONTEXT after ### to respond.

            [EXAMPLE 1]
            [Questions]:
            1. Does the context mention guests are agreeing they can't cancel? 
            2. Does the context mention the guest can call the flexpay team for assistance?
            3. Does the context mention an acount page followed by a link?
            ************
            [CONTEXT]:
            When canceling Universal Studios Hollywood (USH) FlexPay, guests are agreeing to a contract that states they cannot cancel. 
            However, guests can disable the auto-renewal feature at the end of the initial 12-month term by visiting their FlexPay account page at 
            [ush.recurly.com/account/login](https://ush.recurly.com/account/login). Additionally, guests can call the GCC USH FlexPay team at 1-866-254-8275 for 
            assistance with any changes.
            ************
            Answer:
            1. Does the context mention guests are agreeing they can't cancel? 
            Yes, the context mentions that guests agreed they can not cancel
            2. Does the context mention the guest can call the flexpay team for assistance?
            Yes, the context mentions they can call the flexpay team
            3. Does the context mention an acount page followed by a link?
            Yes, the context mentions a web page and link users can find 

            Total Number of Questions: 3
            Total questions answered with Yes: 3
            ************

            [EXAMPLE 2]
            [Questions]:
            1. Does the context mention that all parks have accessible areas and attractions? 
            2. Does the context mention to consult the riders guide for information?
            ************
            [CONTEXT]:
            - Universal Orlando Resort offers wheelchair accessibility at both Universal Studios Florida and Islands of Adventure
            - Restrooms at both parks have wheelchair-accessible facilities and diaper-changing areas
            - Specific attractions at the parks can accommodate guests with physical disabilities requiring wheelchair assistance
            - Guests can consult the Universal Orlando Riders Guide for attraction requirements and accommodations
            ************
            Answer:
            1. Does the context mention that all parks have accessible areas and attractions? 
            No, the answer does not metion all parks have accessible areas an attractions
            2. Does the context mention to consult the riders guide for information?
            Yes, the answer does mention guests can consult the riders guide

            Total number of Questions: 2
            Total questions answered with Yes: 1
            ************

            Now answer each of the questions from [Questions] given below from the CONTEXT given below.

            ###
            ************
            [BEGIN DATA]
            ************
            [Questions]: {criterion}
            ************
            [CONTEXT]: {generated_answer}
            ************
            Answer:
            ************
            [END DATA]
"""

