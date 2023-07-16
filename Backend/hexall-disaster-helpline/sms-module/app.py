

from bottle import (post, request, response, route, run, )
from twilio import twiml
import json

def send_sms(inp):
    f = open("strings.json")
    data = json.load(f)
    account_sid = data["ids"]["acc_sid"]
    auth_token = data["ids"]["auth_token"]
    client = Client(account_sid, auth_token)
    client.messages.create(to="+918073390419", 
                       from_="+13153874357", 
                       body="Missing search request. Inbound: "+inp)


@route('/')
def check_app():
    # returns a simple string stating the app is working
    return "SMS-Module Initiated."


@post('/twilio')
def inbound_sms():
    twiml_response = twiml.Response()
    # grab message from the request. could also get the "To" and 
    # "From" phone numbers as well from parameters with those names#

    inbound_message = request.forms.get("Body")
    # we can now use the incoming message text in our Python application
    if "SOS" in inbound_message:
        twiml_response.message("SOS Detected. Alerting Authorities.")
    elif "MISSING" in inbound_message:
        twiml_response.message("Search request is sent. We will contact you in case we get any updates.")
        send_sms(inbound_message)
    elif "HELP" in inbound_message:
        twiml_response.message("HELP \nSOS <your-locality> to call for help asap. \nMISSING <person-name> to initiate search for concerned person \nSMS ASAP <your-locality> for emergency call to nearest hospital.")
    elif "ASAP" in inbound_message:
        twiml_response.message("Connecting you to nearest hospital.")
    else:
        twiml_response.message("Your request cannot be identified \nPlease type HELP to see the keywords you can use. \n~NoKeywordException")

    # we return back the mimetype because Twilio needs an XML response
    response.content_type = "application/xml"
    return str(twiml_response)


if __name__ == '__main__':
    run(host='127.0.0.1', port=5000, debug=True, reloader=True)