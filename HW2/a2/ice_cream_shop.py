import logging
from random import randint
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session, request


app = Flask(__name__)

ask = Ask(app, "/")

logging.getLogger("flask_ask").setLevel(logging.DEBUG)


def resolved_values(request):
    """
    Helper function that takes in a request object (received from the ASK console)
    and extracts the slot values that were successfuly identified.

    Returns a dictionary of slot name to slot value, in which unresolved values will be populated as None.

    You shouldn't need to change this function, but you can if you think you can improve it.

    THIS FUNCTION SHOULD ONLY BE USED TO EXTRACT VALUES FOR CUSTOM SLOT TYPES!
    If you are using only built-in slot types, you do not need to call it.

    Example usage, to print all the slot values received:

    resolved_vals = resolved_values(request)
    txt = ""
    for key, val in resolved_vals.iteritems():
        txt += "\n{}:{}".format(key, val)


    Args:
        request: request JSON
    
    Returns:
        {intent_slot_name: resolved_value}
    
    Source: https://stackoverflow.com/questions/50349084/alexa-entity-resolution-on-flask-ask-not-handling-synonyms
    """
    print("="*20)
    slots = request["intent"]["slots"]
    slot_names = slots.keys()

    resolved_vals = {}

    for slot_name in slot_names: # iterate over the slot names
        print(slot_name)
        slot = slots[slot_name]

        if "resolutions" in slot:
            slot = slot["resolutions"]["resolutionsPerAuthority"][0]
            slot_status = slot["status"]["code"]
            print(slot_status)
            if slot_status == "ER_SUCCESS_MATCH": # only extract the slot values that were identified as successful
                resolved_val = slot["values"][0]["value"]["name"]
                resolved_vals[slot_name] = resolved_val
            else:
                resolved_vals[slot_name] = None
        else:  # No value found for this slot value
            resolved_vals[slot_name] = None

    print("="*20)
    return resolved_vals

@ask.launch
def new_customer():
    """
    This is one of the request handlers for the skill.
    
    Its name "@ask.launch" means that it will be called once the user opens your skill (e.g. open ice cream shop).

    For now, this function simply gives a return message back to the user as a question. 
    You are free to change it later if you see it fit.

    Note: you always need to wrap a message in question() or statement(), and return it
    If you return a question(), Alexa will expect a response. 
    If you return a statement(), the skill will terminate.
    """
    welcome_msg = render_template('welcome')
    return question(welcome_msg)


@ask.intent("UsernameIntent", convert={'username': str})
def process_order(username):
    """
    This is the request handlers for the UsernameIntent you wrote earlier.
    Pay attention to the syntax of the decorator above: we included the name of the intent, and asked flask to convert the username to a string.
    Notice also how this function uses the resolved_values (using the inherited request object) function to extract the username.
    """
    if username is None or usename=="":
        hello_msg = render_template('hello_no_name', username=username)
    else:
        hello_msg = render_template('hello', username=username)
    return statement(hello_msg)


@ask.intent("AMAZON.FallbackIntent")
def process_order():
    """
    This is the fallback handler.
    
    If Alexa is unsure about the order placed, this function will be called.
    """
    return question("Can I start an ice cream for you?")


if __name__ == '__main__':
    app.run(debug=True)
